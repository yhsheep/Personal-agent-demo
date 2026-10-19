"""RAG 索引与检索服务（pgvector / SQLite 双后端）。"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Paper
from app.services.chunker import chunk_pages
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.utils.text import expand_concepts

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-.]{1,}")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


@dataclass
class IndexResult:
    paper_id: str
    chunk_count: int
    embedding_model: str
    elapsed_ms: int


def _query_terms(query: str) -> dict[str, float]:
    q = (query or "").lower()
    terms: dict[str, float] = {}
    for word in _WORD_RE.findall(q):
        if len(word) > 2:
            key = word.lower()
            terms[key] = max(terms.get(key, 0.0), 1.0)
    for run in _CJK_RUN_RE.findall(q):
        if len(run) <= 8:
            terms[run] = max(terms.get(run, 0.0), 0.45)
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            terms[gram] = max(terms.get(gram, 0.0), 0.22)
    for concept in expand_concepts(q):
        key = concept.lower()
        terms[key] = max(terms.get(key, 0.0), 1.45)
    return terms


def _lexical_overlap(
    query_terms: dict[str, float],
    content: str,
    idf: dict[str, float] | None = None,
) -> float:
    if not query_terms:
        return 0.0
    text = (content or "").lower()
    has_cjk = bool(_CJK_RUN_RE.search(text))
    # 英文论文中没有中文词，不应让中文二元组拉低覆盖率。
    usable = {
        term: weight
        for term, weight in query_terms.items()
        if has_cjk or not _CJK_RUN_RE.search(term)
    }
    if not usable:
        return 0.0
    weighted = {
        term: weight * (idf.get(term, 1.0) if idf else 1.0)
        for term, weight in usable.items()
    }
    hits = sum(weight for term, weight in weighted.items() if term in text)
    return hits / max(1.0, sum(weighted.values()))


def _document_frequencies(
    query_terms: dict[str, float], documents: list[str]
) -> dict[str, float]:
    """在候选集上近似 IDF，强化 limitation / external 这类稀有意图词。"""
    n = max(1, len(documents))
    df = {term: 0 for term in query_terms}
    for doc in documents:
        text = (doc or "").lower()
        for term in query_terms:
            if term in text:
                df[term] += 1
    return {
        term: math.log((n + 2) / (count + 1)) + 0.35
        for term, count in df.items()
    }


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def _row_to_chunk(row: Any) -> dict[str, Any]:
    mapping = dict(row._mapping)
    return {
        "chunk_id": mapping["chunk_id"],
        "paper_id": mapping["paper_id"],
        "paper_title": mapping["paper_title"],
        "paper_keywords": mapping.get("paper_keywords") or [],
        "authors": mapping.get("authors") or [],
        "year": mapping.get("year"),
        "venue": mapping.get("venue") or "",
        "doi": mapping.get("doi") or "",
        "url": mapping.get("url") or "",
        "section": mapping.get("section") or "",
        "page_start": mapping.get("page_start") or 1,
        "page_end": mapping.get("page_end") or 1,
        "content": mapping.get("content") or "",
        "vector_score": float(mapping.get("vector_score") or 0.0),
    }


class RAGService:
    def __init__(self, embedder: EmbeddingService | None = None) -> None:
        self.embedder = embedder or get_embedding_service()

    def index_paper(
        self,
        session: Session,
        paper: Paper,
        pages: list[dict[str, Any]],
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        replace: bool = True,
    ) -> IndexResult:
        started = time.perf_counter()
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size or settings.chunk_size,
            overlap=overlap if overlap is not None else settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("PDF 中未生成有效文本块")
        # 元数据仅参与检索重排，不写入向量文本，避免结论/作者页因标题重复而抢占排名。
        vectors = self.embedder.embed_documents([c.content for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("向量数量与文本块数量不一致")

        if replace:
            session.execute(delete(Chunk).where(Chunk.paper_id == paper.id))
        for chunk, vec in zip(chunks, vectors, strict=True):
            session.add(
                Chunk(
                    paper_id=paper.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    token_count=chunk.token_count,
                    char_count=chunk.char_count,
                    embedding=vec,
                    embedding_model=self.embedder.model_name,
                )
            )
        paper.chunk_count = len(chunks)
        paper.parse_status = "indexed"
        paper.indexed_at = datetime.now(timezone.utc)
        paper.error_message = ""
        paper.updated_at = datetime.now(timezone.utc)
        session.flush()
        return IndexResult(
            paper_id=paper.id,
            chunk_count=len(chunks),
            embedding_model=self.embedder.model_name,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def search(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
        *,
        paper_ids: list[str] | None = None,
        min_score: float | None = None,
        hybrid: bool = True,
        max_per_paper: int = 3,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        k = max(1, min(int(top_k or settings.top_k), 30))
        qvec = self.embedder.embed_query(query)
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            candidates = self._search_postgres(session, qvec, k * 5, paper_ids)
        else:
            candidates = self._search_sqlite(session, qvec, paper_ids)

        terms = _query_terms(query)
        local = self.embedder.is_local
        corpus_docs = [
            f"{c.get('paper_title') or ''} {c.get('paper_keywords') or ''} {c.get('content') or ''}"
            for c in candidates
        ]
        idf = _document_frequencies(terms, corpus_docs)
        for item in candidates:
            lexical = _lexical_overlap(terms, item["content"], idf)
            title_hit = _lexical_overlap(
                terms,
                f"{item.get('paper_title') or ''} {item.get('paper_keywords') or ''}",
                idf,
            )
            item["lexical_score"] = round(lexical, 6)
            item["title_score"] = round(title_hit, 6)
            base = item["vector_score"]
            if hybrid:
                if local:
                    score = 0.45 * base + 0.55 * lexical + 0.10 * title_hit
                else:
                    score = 0.82 * base + 0.18 * lexical + 0.04 * title_hit
                item["score"] = round(score, 6)
            else:
                item["score"] = base
        threshold = settings.min_score if min_score is None else min_score
        candidates = [c for c in candidates if c["score"] >= threshold]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 单篇论文最多保留 max_per_paper 个块，避免上下文被同一篇论文垄断
        picked: list[dict[str, Any]] = []
        per_paper: dict[str, int] = {}
        for item in candidates:
            count = per_paper.get(item["paper_id"], 0)
            if count >= max_per_paper:
                continue
            per_paper[item["paper_id"]] = count + 1
            item["rank"] = len(picked) + 1
            picked.append(item)
            if len(picked) >= k:
                break
        return picked

    def _search_postgres(
        self,
        session: Session,
        qvec: list[float],
        limit: int,
        paper_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        where = "c.embedding IS NOT NULL"
        params: dict[str, Any] = {"qvec": _vector_literal(qvec), "limit": limit}
        stmt = text(
            f"""
            SELECT c.id AS chunk_id, c.paper_id, c.content, c.section,
                   c.page_start, c.page_end,
                   p.title AS paper_title, p.authors, p.year, p.venue, p.doi, p.url,
                   p.keywords AS paper_keywords,
                   1 - (c.embedding <=> CAST(:qvec AS vector)) AS vector_score
            FROM rag_chunks c
            JOIN papers p ON p.id = c.paper_id
            WHERE {where}
            {"AND c.paper_id IN :paper_ids" if paper_ids else ""}
            ORDER BY c.embedding <=> CAST(:qvec AS vector)
            LIMIT :limit
            """
        )
        if paper_ids:
            stmt = stmt.bindparams(bindparam("paper_ids", expanding=True))
            params["paper_ids"] = paper_ids
        rows = session.execute(stmt, params).all()
        out = [_row_to_chunk(row) for row in rows]
        # pgvector 的余弦距离转相似度，数值误差归一到 [-1, 1]
        for item in out:
            item["vector_score"] = max(-1.0, min(1.0, item["vector_score"]))
        return out

    def _search_sqlite(
        self,
        session: Session,
        qvec: list[float],
        paper_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(Chunk, Paper)
            .join(Paper, Paper.id == Chunk.paper_id)
            .where(Chunk.embedding.is_not(None))
        )
        if paper_ids:
            stmt = stmt.where(Chunk.paper_id.in_(paper_ids))
        rows = session.execute(stmt).all()
        if not rows:
            return []
        q = np.asarray(qvec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q)) or 1.0
        out: list[dict[str, Any]] = []
        for chunk, paper in rows:
            vec = chunk.embedding or []
            if len(vec) != len(qvec):
                continue
            arr = np.asarray(vec, dtype=np.float32)
            score = float(np.dot(q, arr) / (q_norm * (float(np.linalg.norm(arr)) or 1.0)))
            out.append(
                {
                    "chunk_id": chunk.id,
                    "paper_id": paper.id,
                    "paper_title": paper.title,
                    "paper_keywords": paper.keywords or [],
                    "authors": paper.authors or [],
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "url": paper.url,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "content": chunk.content,
                    "vector_score": score,
                }
            )
        out.sort(key=lambda x: x["vector_score"], reverse=True)
        return out

    def context_chunks(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
        *,
        max_chars: int = 9000,
    ) -> list[dict[str, Any]]:
        chunks = self.search(session, query, top_k)
        picked: list[dict[str, Any]] = []
        used = 0
        for item in chunks:
            if picked and used + len(item["content"]) > max_chars:
                break
            picked.append(item)
            used += len(item["content"])
        return picked


def citation_list(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        citations.append(
            {
                "index": i,
                "chunk_id": c.get("chunk_id"),
                "paper_id": c.get("paper_id"),
                "paper_title": c.get("paper_title"),
                "authors": c.get("authors") or [],
                "year": c.get("year"),
                "venue": c.get("venue") or "",
                "section": c.get("section") or "",
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "score": c.get("score", c.get("vector_score", 0.0)),
                "snippet": (c.get("content") or "")[:420],
            }
        )
    return citations


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
