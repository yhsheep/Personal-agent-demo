"""SQLAlchemy ORM 模型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EmbeddingType(TypeDecorator):
    """PostgreSQL 使用 pgvector；SQLite 使用 JSON 保存向量。"""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_result_value(self, value, dialect):  # type: ignore[no-untyped-def]
        if value is None:
            return None
        return [float(x) for x in value]


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(600), nullable=False, index=True)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    venue: Mapped[str] = mapped_column(String(300), default="")
    doi: Mapped[str | None] = mapped_column(String(300), index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(100), index=True)
    url: Mapped[str] = mapped_column(Text, default="")
    pdf_url: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(50), default="upload")
    language: Mapped[str] = mapped_column(String(20), default="en")
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)

    raw_text: Mapped[str | None] = mapped_column(Text)
    page_texts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    pdf_path: Mapped[str | None] = mapped_column(Text)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)

    parse_status: Mapped[str] = mapped_column(String(30), default="metadata")
    error_message: Mapped[str] = mapped_column(Text, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    paper_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(String(300), default="")
    page_start: Mapped[int] = mapped_column(Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(
        EmbeddingType(settings.embedding_dim)
    )
    embedding_model: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(300), default="新的科研对话")
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(300), default="评测任务")
    kind: Mapped[str] = mapped_column(String(50), default="agent")
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    dataset: Mapped[str] = mapped_column(String(300), default="")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    details: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error_message: Mapped[str] = mapped_column(Text, default="")
    llm_provider: Mapped[str] = mapped_column(String(100), default="")
    embedding_model: Mapped[str] = mapped_column(String(200), default="")
    case_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchRecord(Base):
    """论文外部检索缓存/发现记录。"""

    __tablename__ = "search_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    query: Mapped[str] = mapped_column(String(600), index=True)
    source: Mapped[str] = mapped_column(String(50), default="")
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    is_imported: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExperimentResult(Base):
    """参数实验结果（Chunk / Top-K / Prompt）。"""

    __tablename__ = "experiment_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), default="参数实验")
    kind: Mapped[str] = mapped_column(String(50), default="retrieval_grid")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    table: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    best_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
