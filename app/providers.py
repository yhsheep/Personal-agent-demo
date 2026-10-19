"""模型服务预设：国内常用的 OpenAI 兼容接口。"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    is_mock: bool = False

    @property
    def label(self) -> str:
        if self.is_mock:
            return "离线演示模型"
        return f"{self.provider} / {self.model}"


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    dimension: int
    is_local: bool = False

    @property
    def label(self) -> str:
        if self.is_local:
            return f"local-hash / {self.dimension}d"
        return f"{self.provider} / {self.model}"


LLM_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "label": "DeepSeek",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "label": "通义千问 / 百炼",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "label": "硅基流动",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "label": "Kimi / 月之暗面",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "label": "智谱 GLM",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "label": "OpenAI 兼容",
    },
}

EMBEDDING_PRESETS: dict[str, dict[str, str | int]] = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "BAAI/bge-m3",
        "dim": 1024,
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "text-embedding-v4",
        "dim": 1024,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "dim": 1024,
    },
}

_PROVIDER_KEYS = {
    "deepseek": "deepseek_api_key",
    "dashscope": "dashscope_api_key",
    "siliconflow": "siliconflow_api_key",
    "moonshot": "moonshot_api_key",
    "zhipu": "zhipu_api_key",
}


def resolve_llm_config(settings: Settings) -> LLMConfig:
    """解析 LLM 配置。

    优先级：显式 LLM_API_KEY > 单供应商 Key > mock。
    LLM_PROVIDER=auto 时按 deepseek → dashscope → siliconflow → moonshot → zhipu 检测。
    """
    provider = (settings.llm_provider or "auto").strip().lower()

    if provider == "mock":
        return LLMConfig("mock", "", "", "wenlan-demo", True)

    if settings.llm_api_key:
        preset = LLM_PRESETS.get(provider if provider != "auto" else "openai")
        if provider == "auto":
            preset = {"base_url": "", "model": "gpt-4o-mini"}
        return LLMConfig(
            provider=provider if provider != "auto" else "openai",
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or str(preset["base_url"]),
            model=settings.llm_model or str(preset["model"]),
        )

    if provider != "auto":
        key = getattr(settings, _PROVIDER_KEYS.get(provider, ""), "")
        if key:
            preset = LLM_PRESETS[provider]
            return LLMConfig(
                provider=provider,
                api_key=key,
                base_url=settings.llm_base_url or str(preset["base_url"]),
                model=settings.llm_model or str(preset["model"]),
            )
        # 显式指定但未配置 Key：进入 mock，避免应用启动失败。
        return LLMConfig("mock", "", "", "wenlan-demo", True)

    for name in ("deepseek", "dashscope", "siliconflow", "moonshot", "zhipu"):
        key = getattr(settings, _PROVIDER_KEYS[name], "")
        if key:
            preset = LLM_PRESETS[name]
            return LLMConfig(
                provider=name,
                api_key=key,
                base_url=settings.llm_base_url or str(preset["base_url"]),
                model=settings.llm_model or str(preset["model"]),
            )

    return LLMConfig("mock", "", "", "wenlan-demo", True)


def resolve_embedding_config(settings: Settings) -> EmbeddingConfig:
    provider = (settings.embedding_provider or "auto").strip().lower()
    if provider == "local":
        return EmbeddingConfig("local", "", "", "local-hash-ngram", settings.embedding_dim, True)

    if settings.embedding_api_key:
        if provider == "auto":
            provider = "openai"
        preset = EMBEDDING_PRESETS.get(provider, EMBEDDING_PRESETS["openai"])
        return EmbeddingConfig(
            provider=provider,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or str(preset["base_url"]),
            model=settings.embedding_model or str(preset["model"]),
            dimension=int(preset.get("dim", settings.embedding_dim)),
        )

    if provider != "auto":
        key_name = _PROVIDER_KEYS.get(provider, "")
        key = getattr(settings, key_name, "")
        if key:
            preset = EMBEDDING_PRESETS[provider]
            return EmbeddingConfig(
                provider=provider,
                api_key=key,
                base_url=settings.embedding_base_url or str(preset["base_url"]),
                model=settings.embedding_model or str(preset["model"]),
                dimension=int(preset.get("dim", settings.embedding_dim)),
            )

    if settings.embedding_provider.strip().lower() == "auto":
        for name in ("siliconflow", "dashscope"):
            key = getattr(settings, _PROVIDER_KEYS[name], "")
            if key:
                preset = EMBEDDING_PRESETS[name]
                return EmbeddingConfig(
                    provider=name,
                    api_key=key,
                    base_url=settings.embedding_base_url or str(preset["base_url"]),
                    model=settings.embedding_model or str(preset["model"]),
                    dimension=int(preset.get("dim", settings.embedding_dim)),
                )

    # 没有任何远端 Key → 本地哈希向量。无需额外配置即可运行。
    return EmbeddingConfig("local", "", "", "local-hash-ngram", settings.embedding_dim, True)
