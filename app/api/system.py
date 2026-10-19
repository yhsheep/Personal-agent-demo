"""系统状态与健康检查。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__
from app.config import settings
from app.db import db_status, get_db
from app.models import Chunk, Conversation, EvaluationRun, Paper
from app.services.embeddings import get_embedding_service
from app.services.llm import get_llm_service
from app.services.prompts import PROMPT_VARIANTS

router = APIRouter(tags=["system"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    db_info = db_status()
    llm = get_llm_service().status()
    embedding = get_embedding_service().status()
    return {
        "status": "ok" if db_info.get("ok") else "degraded",
        "version": __version__,
        "database": db_info,
        "llm": llm,
        "embedding": embedding,
        "demo_mode": llm.get("is_mock", False),
    }


@router.get("/system/info")
def info(db: Session = Depends(get_db)) -> dict:
    llm = get_llm_service().status()
    embedding = get_embedding_service().status()
    counts = {
        "papers": db.scalar(select(func.count(Paper.id))) or 0,
        "chunks": db.scalar(select(func.count(Chunk.id))) or 0,
        "conversations": db.scalar(select(func.count(Conversation.id))) or 0,
        "evaluations": db.scalar(select(func.count(EvaluationRun.id))) or 0,
    }
    return {
        "app_name": settings.app_name,
        "version": __version__,
        "environment": settings.app_env,
        "demo_mode": llm.get("is_mock", False),
        "llm": llm,
        "embedding": embedding,
        "database": db_status(),
        "counts": counts,
        "limits": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "max_upload_mb": settings.max_upload_mb,
            "max_pdf_pages": settings.max_pdf_pages,
            "agent_max_tool_rounds": settings.agent_max_tool_rounds,
            "agent_online_search": settings.agent_online_search,
        },
        "features": {
            "tool_calling": True,
            "pgvector": "postgresql" in str(db_status().get("dialect")),
            "multi_turn": True,
            "evaluation": True,
        },
        "prompt_variants": [
            {"key": k, "label": v["label"]} for k, v in PROMPT_VARIANTS.items()
        ],
    }


@router.get("/system/prompts")
def prompts() -> dict:
    return {
        "variants": [
            {
                "key": key,
                "label": value["label"],
                "temperature": value["temperature"],
                "description": value["system"].split("额外要求：")[-1][:180],
            }
            for key, value in PROMPT_VARIANTS.items()
        ],
        "default": "citation_strict",
    }
