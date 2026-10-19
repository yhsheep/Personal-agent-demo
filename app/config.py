"""集中配置。所有配置均支持环境变量与项目根目录 .env。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "文澜 WenLan"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # LLM
    llm_provider: str = "auto"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0

    deepseek_api_key: str = ""
    dashscope_api_key: str = ""
    siliconflow_api_key: str = ""
    moonshot_api_key: str = ""
    zhipu_api_key: str = ""

    # Embedding
    embedding_provider: str = "auto"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # Database
    database_url: str = ""
    use_embedded_postgres: bool = True
    pgdata_dir: str = "data/pgdata"
    sqlite_fallback: bool = True

    # RAG
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 6
    min_score: float = 0.0
    max_context_chunks: int = 8
    max_upload_mb: int = 30
    max_pdf_pages: int = 300

    # Agent
    agent_max_tool_rounds: int = 4
    agent_history_turns: int = 8
    agent_online_search: bool = False
    openalex_mailto: str = ""
    http_proxy_url: str = ""

    # Evaluation
    eval_dataset: str = "data/eval/dataset.jsonl"
    eval_max_cases: int = 30

    # 多来源检索
    paper_sources: str = "local,openalex,arxiv"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 600)
        if v >= size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v < 64 or v > 8192:
            raise ValueError("EMBEDDING_DIM 建议在 64~8192 之间")
        return v

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def uploads_dir(self) -> Path:
        return DATA_DIR / "uploads"

    @property
    def exports_dir(self) -> Path:
        return DATA_DIR / "exports"

    @property
    def logs_dir(self) -> Path:
        return DATA_DIR / "logs"

    @property
    def pgdata_path(self) -> Path:
        p = Path(self.pgdata_dir)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def eval_dataset_path(self) -> Path:
        p = Path(self.eval_dataset)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def is_demo_llm(self) -> bool:
        return self.llm_provider.strip().lower() == "mock"

    def ensure_dirs(self) -> None:
        for p in (DATA_DIR, self.uploads_dir, self.exports_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "seed" / "pdfs").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
