"""评测集与实验 API。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import EvaluationRun
from app.schemas import EvalRunRequest
from app.services.evaluation import (
    get_evaluation_service,
    load_dataset,
    run_to_dict,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluations"])
evaluation = get_evaluation_service()


@router.get("/evaluations/dataset")
def dataset_info(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    try:
        cases = load_dataset(limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    by_type: dict[str, int] = {}
    for case in cases:
        by_type[case.get("type", "unknown")] = by_type.get(case.get("type", "unknown"), 0) + 1
    return {
        "path": str(settings.eval_dataset_path),
        "count": len(cases),
        "by_type": by_type,
        "cases": [
            {
                "id": c["id"],
                "type": c.get("type"),
                "difficulty": c.get("difficulty"),
                "question": c["question"],
                "expected_tools": c.get("expected_tools", []),
                "relevant_papers": c.get("relevant_papers", []),
            }
            for c in cases
        ],
    }


@router.post("/evaluations/run")
def run_evaluation(body: EvalRunRequest, db: Session = Depends(get_db)) -> dict:
    try:
        run = evaluation.start_run(
            db,
            kind=body.kind,
            name=body.name,
            limit=body.limit,
            top_k=body.top_k,
            prompt_variant=body.prompt_variant,
            chunk_sizes=body.chunk_sizes,
            overlaps=body.overlaps,
            top_ks=body.top_ks,
        )
        return {"run": run_to_dict(run)}
    except Exception as exc:
        db.rollback()
        logger.exception("启动评测失败")
        raise HTTPException(status_code=500, detail=f"启动评测失败：{exc}") from exc


@router.get("/evaluations/runs")
def list_runs(limit: int = Query(default=30, ge=1, le=100), db: Session = Depends(get_db)) -> dict:
    runs = db.scalars(
        select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
    ).all()
    return {"items": [run_to_dict(r, include_details=False) for r in runs]}


@router.get("/evaluations/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return {"run": run_to_dict(run)}


@router.delete("/evaluations/runs/{run_id}")
def delete_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    db.delete(run)
    db.commit()
    return {"deleted": True}
