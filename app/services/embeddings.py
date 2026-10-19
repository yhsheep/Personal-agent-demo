"""Embedding 服务：远端 OpenAI 兼容接口 + 可离线运行的哈希 n-gram 向量。"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
import unicodedata
from collections.abc import Iterable

import httpx
import numpy as np

from app.config import settings
from app.providers import EmbeddingConfig, resolve_embedding_config
from app.utils.text import expand_concepts

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-.]{1,}")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


class EmbeddingError(RuntimeError):
    pass


class LocalHashEmbedding:
    """零依赖、离线可用的 signed feature hashing 向量。

    采用中英文词 + 中文 2/3-gram + 中英术语桥接，适合小规模演示与回归测试。
    它不是语义模型，长期使用建议配置 bge-m3 / text-embedding-v4。
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        text = unicodedata.normalize("NFKC", text or "").lower()
        feats: list[tuple[str, float]] = []
        for w in _WORD_RE.findall(text):
            feats.append((f"w:{w}", 1.0))
            if len(w) >= 5:
                for i in range(len(w) - 2):
                    feats.append((f"g:{w[i:i + 3]}", 0.28))
        for run in _CJK_RUN_RE.findall(text):
            for ch in run:
                feats.append((f"c:{ch}", 0.35))
            for n, weight in ((2, 1.0), (3, 0.75)):
                if len(run) >= n:
                    for i in range(len(run) - n + 1):
                        feats.append((f"n{n}:{run[i:i + n]}", weight))
        for concept in expand_concepts(text):
            # 同时写入概念命名空间与普通词命名空间，保证中文查询与英文原文
            # 在 "generator"/"generator" 这类桥接词上真正共享特征。
            feats.append((f"x:{concept}", 0.9))
            feats.append((f"w:{concept}", 0.55))
        if not feats and text.strip():
            feats.append((f"raw:{text.strip()[:40]}", 1.0))
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, float] = {}
        for feature, weight in self._features(text):
            counts[feature] = counts.get(feature, 0.0) + weight
        for feature, tf in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log1p(tf))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class RemoteEmbedding:
    def __init__(self, config: EmbeddingConfig, timeout: float = 60.0) -> None:
        self.config = config
        self.timeout = timeout

    def _payload(self, texts: list[str]) -> dict:
        payload: dict = {"model": self.config.model, "input": texts}
        # bge-m3 等硅基流动模型不接受 dimensions 字段。
        if self.config.provider in {"dashscope", "openai"}:
            payload["dimensions"] = self.config.dimension
        return payload

    def _request(self, texts: list[str]) -> list[list[float]]:
        url = self.config.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=self._payload(texts), headers=headers)
                if resp.status_code >= 400:
                    detail = resp.text[:500]
                    raise EmbeddingError(
                        f"Embedding API {resp.status_code}: {detail}"
                    )
                data = resp.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                vectors = [item["embedding"] for item in items]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Embedding 返回数量与请求不一致")
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
        raise EmbeddingError(f"Embedding API 调用失败：{last_error}")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch = max(1, settings.embedding_batch_size)
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            vectors = self._request(chunk)
            for vec in vectors:
                if len(vec) != self.config.dimension:
                    raise EmbeddingError(
                        f"Embedding 维度不匹配：期望 {self.config.dimension}，实际 {len(vec)}。"
                        "请调整 EMBEDDING_DIM 并重建向量索引。"
                    )
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    arr = arr / norm
                out.append(arr.tolist())
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or resolve_embedding_config(settings)
        if self.config.is_local:
            self.impl: LocalHashEmbedding | RemoteEmbedding = LocalHashEmbedding(
                self.config.dimension
            )
        else:
            self.impl = RemoteEmbedding(self.config)

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def is_local(self) -> bool:
        return self.config.is_local

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        items = [t or "" for t in texts]
        if not items:
            return []
        return self.impl.embed_many(items)

    def embed_query(self, text: str) -> list[float]:
        return self.impl.embed_one(text or "")

    def status(self) -> dict:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "dimension": self.config.dimension,
            "local": self.config.is_local,
            "label": self.config.label,
        }


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def reset_embedding_service() -> None:
    global _embedding_service
    _embedding_service = None
