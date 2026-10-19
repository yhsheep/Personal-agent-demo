"""API 请求 / 响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.services.paper_sources import PaperCandidate
from app.utils.text import sanitize_text


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=20)
    prompt_variant: str = "citation_strict"

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = sanitize_text(v).strip()
        if not v:
            raise ValueError("消息不能为空")
        return v


class ConversationCreate(BaseModel):
    title: str = Field(default="新的科研对话", max_length=120)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return sanitize_text(v).strip() or "新的科研对话"


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        v = sanitize_text(v).strip()
        if not v:
            raise ValueError("标题不能为空")
        return v


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def _clean_query(cls, v: str) -> str:
        v = sanitize_text(v).strip()
        if not v:
            raise ValueError("检索词不能为空")
        return v
    top_k: int = Field(default=6, ge=1, le=30)
    paper_ids: list[str] | None = None
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class ReindexRequest(BaseModel):
    chunk_size: int | None = Field(default=None, ge=100, le=2000)
    overlap: int | None = Field(default=None, ge=0, le=1000)


class CandidateImport(BaseModel):
    title: str = Field(min_length=2, max_length=600)
    authors: list[str] = []
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = "import"
    external_id: str = ""
    citation_count: int = 0
    fetch_pdf: bool = True

    def to_candidate(self) -> PaperCandidate:
        return PaperCandidate(
            title=self.title,
            authors=self.authors,
            abstract=self.abstract,
            year=self.year,
            venue=self.venue,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            url=self.url,
            pdf_url=self.pdf_url,
            source=self.source,
            external_id=self.external_id,
            citation_count=self.citation_count,
        )


class EvalRunRequest(BaseModel):
    kind: Literal["full", "retrieval", "agent", "retrieval_grid", "prompt_grid"] = "full"
    name: str = ""
    limit: int | None = Field(default=None, ge=1, le=100)
    top_k: int = Field(default=6, ge=1, le=20)
    prompt_variant: str = "citation_strict"
    chunk_sizes: list[int] = [300, 500, 700]
    overlaps: list[int] = [50, 100]
    top_ks: list[int] = [3, 5, 8]

    @field_validator("chunk_sizes")
    @classmethod
    def _chunk_sizes(cls, v: list[int]) -> list[int]:
        return sorted({max(100, min(2000, int(x))) for x in v})[:5]

    @field_validator("overlaps")
    @classmethod
    def _overlaps(cls, v: list[int]) -> list[int]:
        return sorted({max(0, min(500, int(x))) for x in v})[:4]

    @field_validator("top_ks")
    @classmethod
    def _top_ks(cls, v: list[int]) -> list[int]:
        return sorted({max(1, min(20, int(x))) for x in v})[:5]
