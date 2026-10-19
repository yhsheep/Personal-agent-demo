"""离线可复现的 Agent / RAG 评测与参数实验。

指标：
- Recall@K / MRR / nDCG@K：检索质量（论文级）
- Tool Calling Accuracy：必需工具是否被调用（coverage），并报告 exact match 与 F1
- Task Success Rate：关键词事实 + 引用落地双重校验
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import EvaluationRun, Paper
from app.services.agent import AgentService, get_agent_service
from app.services.chunker import chunk_pages
from app.services.embeddings import get_embedding_service
from app.services.prompts import PROMPT_VARIANTS
from app.services.rag import (
    RAGService,
    _document_frequencies,
    _lexical_overlap,
    _query_terms,
)

logger = logging.getLogger(__name__)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (title or "").lower())


def load_dataset(path: str | Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else settings.eval_dataset_path
    if not p.exists():
        raise FileNotFoundError(f"评测集不存在：{p}")
    cases: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    if limit:
        cases = cases[:limit]
    return cases


def _resolve_relevant_ids(session: Session, cases: list[dict[str, Any]]) -> dict[str, list[str]]:
    papers = list(session.scalars(select(Paper)).all())
    by_norm = {_norm_title(p.title): p.id for p in papers}
    mapping: dict[str, list[str]] = {}
    for case in cases:
        ids: list[str] = []
        for title in case.get("relevant_papers") or []:
            key = _norm_title(title)
            pid = by_norm.get(key)
            if not pid:
                # 允许标题前缀/子串匹配
                for norm, candidate_id in by_norm.items():
                    if key and (key in norm or norm in key):
                        pid = candidate_id
                        break
            if pid:
                ids.append(pid)
        mapping[case["id"]] = list(dict.fromkeys(ids))
    return mapping


def _keyword_ok(answer: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    text = (answer or "").lower()
    missing: list[str] = []
    for group in groups:
        if not any(str(term).lower() in text for term in group):
            missing.append(" / ".join(map(str, group)))
    return not missing, missing


def _paper_level_metrics(
    ranked_paper_ids: list[str], relevant: set[str], k: int
) -> dict[str, float]:
    if not relevant:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0}
    seen: set[str] = set()
    first_rank: int | None = None
    dcg = 0.0
    for rank, pid in enumerate(ranked_paper_ids[:k], start=1):
        if pid in seen:
            continue
        seen.add(pid)
        if pid in relevant:
            if first_rank is None:
                first_rank = rank
            dcg += 1.0 / math.log2(rank + 1)
    hits = len(seen & relevant)
    recall = hits / len(relevant)
    mrr = 1.0 / first_rank if first_rank else 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    precision = hits / max(1, len(seen))
    return {
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "precision": round(precision, 4),
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


class EvaluationService:
    def __init__(self, agent: AgentService | None = None) -> None:
        self.agent = agent or get_agent_service()
        self.rag = RAGService()

    # ------------------------------------------------------------------
    def start_run(
        self,
        session: Session,
        *,
        kind: str = "full",
        name: str = "",
        limit: int | None = None,
        top_k: int = 6,
        prompt_variant: str = "citation_strict",
        chunk_sizes: list[int] | None = None,
        overlaps: list[int] | None = None,
        top_ks: list[int] | None = None,
    ) -> EvaluationRun:
        if kind not in {"full", "retrieval", "agent", "retrieval_grid", "prompt_grid"}:
            raise ValueError(f"不支持的评测类型：{kind}")
        config = {
            "limit": limit,
            "top_k": top_k,
            "prompt_variant": prompt_variant,
            "chunk_sizes": chunk_sizes or [300, 500, 700],
            "overlaps": overlaps or [50, 100],
            "top_ks": top_ks or [3, 5, 8],
        }
        run = EvaluationRun(
            name=name or {
                "full": "完整评测",
                "retrieval": "检索评测",
                "agent": "Agent 评测",
                "retrieval_grid": "Chunk × Top-K 网格实验",
                "prompt_grid": "Prompt 对比实验",
            }.get(kind, "评测任务"),
            kind=kind,
            status="queued",
            dataset=settings.eval_dataset_path.name,
            config=config,
            llm_provider=self.agent.llm.status().get("label", ""),
            embedding_model=get_embedding_service().model_name,
        )
        session.add(run)
        session.commit()
        thread = threading.Thread(
            target=self._thread_entry,
            args=(run.id, kind, config),
            daemon=True,
            name=f"wenlan-eval-{run.id[:8]}",
        )
        thread.start()
        return run

    def _thread_entry(self, run_id: str, kind: str, config: dict[str, Any]) -> None:
        with session_scope() as session:
            try:
                self.execute_run(session, run_id, kind, config)
            except Exception as exc:
                logger.exception("评测执行失败：%s", exc)
                session.rollback()
                run = session.get(EvaluationRun, run_id)
                if run:
                    run.status = "failed"
                    run.error_message = str(exc)
                    run.completed_at = datetime.now(timezone.utc)
                    session.commit()

    def execute_run(
        self,
        session: Session,
        run_id: str,
        kind: str,
        config: dict[str, Any],
    ) -> None:
        run = session.get(EvaluationRun, run_id)
        if not run:
            return
        run.status = "running"
        session.commit()
        started = time.perf_counter()

        if kind == "retrieval_grid":
            self._run_retrieval_grid(session, run, config)
        elif kind == "prompt_grid":
            self._run_prompt_grid(session, run, config)
        else:
            cases = load_dataset(limit=config.get("limit") or settings.eval_max_cases)
            relevant = _resolve_relevant_ids(session, cases)
            details: list[dict[str, Any]] = []
            metrics: dict[str, Any] = {"case_count": len(cases), "k": config.get("top_k", 6)}
            if kind in {"full", "retrieval"}:
                retrieval_details, retrieval_metrics = self._retrieval_cases(
                    session, cases, relevant, int(config.get("top_k") or 6)
                )
                details.extend(retrieval_details)
                metrics.update(retrieval_metrics)
            if kind in {"full", "agent"}:
                agent_details, agent_metrics = self._agent_cases(
                    session,
                    cases,
                    relevant,
                    top_k=int(config.get("top_k") or 6),
                    prompt_variant=str(config.get("prompt_variant") or "citation_strict"),
                )
                details.extend(agent_details)
                metrics.update(agent_metrics)
            run.details = details
            run.metrics = metrics
            run.case_count = len(cases)
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.metrics["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            session.commit()
            return

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()

    # ------------------------------------------------------------------
    def _retrieval_cases(
        self,
        session: Session,
        cases: list[dict[str, Any]],
        relevant: dict[str, list[str]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for case in cases:
            rel = set(relevant.get(case["id"], []))
            started = time.perf_counter()
            chunks = self.rag.search(session, case["question"], top_k=top_k)
            latency = int((time.perf_counter() - started) * 1000)
            ranked_ids = [c["paper_id"] for c in chunks]
            metrics = _paper_level_metrics(ranked_ids, rel, top_k)
            details.append(
                {
                    "id": case["id"],
                    "stage": "retrieval",
                    "type": case.get("type"),
                    "question": case["question"],
                    "relevant_papers": [
                        self._paper_title(session, pid) for pid in relevant.get(case["id"], [])
                    ],
                    "retrieved": [
                        {
                            "rank": i + 1,
                            "paper_id": c["paper_id"],
                            "title": c["paper_title"],
                            "section": c["section"],
                            "page": c["page_start"],
                            "score": c["score"],
                        }
                        for i, c in enumerate(chunks)
                    ],
                    "recall": metrics["recall"],
                    "mrr": metrics["mrr"],
                    "ndcg": metrics["ndcg"],
                    "precision": metrics["precision"],
                    "hit": metrics["recall"] > 0,
                    "latency_ms": latency,
                }
            )
        aggregate = {
            "retrieval_recall_at_k": _avg([d["recall"] for d in details]),
            "retrieval_mrr": _avg([d["mrr"] for d in details]),
            "retrieval_ndcg_at_k": _avg([d["ndcg"] for d in details]),
            "retrieval_precision_at_k": _avg([d["precision"] for d in details]),
            "retrieval_hit_rate": _avg([1.0 if d["hit"] else 0.0 for d in details]),
            "retrieval_latency_ms": _avg([float(d["latency_ms"]) for d in details]),
        }
        return details, aggregate

    @staticmethod
    def _paper_title(session: Session, paper_id: str) -> str:
        paper = session.get(Paper, paper_id)
        return paper.title if paper else paper_id

    def _agent_cases(
        self,
        session: Session,
        cases: list[dict[str, Any]],
        relevant: dict[str, list[str]],
        *,
        top_k: int,
        prompt_variant: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        details: list[dict[str, Any]] = []
        title_map = {
            p.id: p.title for p in session.scalars(select(Paper)).all()
        }
        for case in cases:
            started = time.perf_counter()
            error = ""
            try:
                result = self.agent.answer_once(
                    session,
                    case["question"],
                    history=case.get("history") or [],
                    top_k=top_k,
                    prompt_variant=prompt_variant,
                )
                answer = result["answer"]
                trace = result["tool_trace"]
                citations = result["citations"]
            except Exception as exc:
                logger.exception("Agent 评测用例失败：%s", case.get("id"))
                answer, trace, citations = "", [], []
                result = {"usage": {}, "latency_ms": 0}
                error = str(exc)
            latency = int((time.perf_counter() - started) * 1000)
            expected = list(case.get("expected_tools") or [])
            actual = [t["name"] for t in trace]
            actual_unique = list(dict.fromkeys(actual))
            hit = set(expected) & set(actual_unique)
            coverage = set(expected).issubset(set(actual_unique))
            precision = len(hit) / len(actual_unique) if actual_unique else (1.0 if not expected else 0.0)
            recall = len(hit) / len(expected) if expected else 1.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            keyword_ok, missing = _keyword_ok(answer, case.get("keyword_groups") or [])
            rel_titles = {
                _norm_title(title_map.get(pid, ""))
                for pid in relevant.get(case["id"], [])
                if pid in title_map
            }
            cited = [c for c in citations if c.get("cited")]
            citation_hit = bool(
                rel_titles
                and any(_norm_title(c.get("paper_title", "")) in rel_titles for c in cited)
            )
            needs_citation = "search_library" in expected
            citation_ok = (not needs_citation) or citation_hit
            refusal = "没有足够证据" in (answer or "") and needs_citation
            success = bool(keyword_ok and citation_ok and not refusal and not error)
            details.append(
                {
                    "id": case["id"],
                    "stage": "agent",
                    "type": case.get("type"),
                    "question": case["question"],
                    "answer": answer,
                    "reference_answer": case.get("reference_answer", ""),
                    "expected_tools": expected,
                    "actual_tools": actual_unique,
                    "tool_coverage": coverage,
                    "tool_exact": set(expected) == set(actual_unique),
                    "tool_precision": round(precision, 4),
                    "tool_recall": round(recall, 4),
                    "tool_f1": round(f1, 4),
                    "keyword_ok": keyword_ok,
                    "missing_keywords": missing,
                    "citation_hit": citation_hit,
                    "citation_count": len(citations),
                    "success": success,
                    "error": error,
                    "latency_ms": latency,
                    "usage": result.get("usage") or {},
                    "tool_trace": trace,
                    "citations": [
                        {
                            "index": c.get("index"),
                            "paper_title": c.get("paper_title"),
                            "page": c.get("page_start"),
                            "cited": c.get("cited"),
                        }
                        for c in citations
                    ],
                }
            )
        aggregate = {
            "task_success_rate": _avg([1.0 if d["success"] else 0.0 for d in details]),
            "tool_calling_accuracy": _avg(
                [1.0 if d["tool_coverage"] else 0.0 for d in details]
            ),
            "tool_exact_match": _avg([1.0 if d["tool_exact"] else 0.0 for d in details]),
            "tool_f1": _avg([float(d["tool_f1"]) for d in details]),
            "citation_hit_rate": _avg(
                [1.0 if d["citation_hit"] else 0.0 for d in details]
            ),
            "agent_latency_ms": _avg([float(d["latency_ms"]) for d in details]),
            "avg_tokens": _avg(
                [float((d.get("usage") or {}).get("total_tokens") or 0) for d in details]
            ),
        }
        return details, aggregate

    # ------------------------------------------------------------------
    def _run_retrieval_grid(
        self, session: Session, run: EvaluationRun, config: dict[str, Any]
    ) -> None:
        cases = load_dataset(limit=config.get("limit") or 20)
        relevant = _resolve_relevant_ids(session, cases)
        papers = [
            p
            for p in session.scalars(select(Paper)).all()
            if p.page_texts
        ]
        if not papers:
            raise RuntimeError("没有可用于实验的 PDF 文档，请先导入示例文献库")
        chunk_sizes = [int(x) for x in config.get("chunk_sizes") or [300, 500, 700]]
        overlaps = [int(x) for x in config.get("overlaps") or [50, 100]]
        top_ks = [int(x) for x in config.get("top_ks") or [3, 5, 8]]
        max_top_k = max(top_ks)
        embedder = get_embedding_service()

        query_vectors: dict[str, list[float]] = {}
        query_terms = {
            c["id"]: _query_terms(c["question"]) for c in cases
        }
        for case in cases:
            query_vectors[case["id"]] = embedder.embed_query(case["question"])

        rows: list[dict[str, Any]] = []
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue
                texts: list[str] = []
                meta: list[dict[str, Any]] = []
                for paper in papers:
                    chunks = chunk_pages(
                        paper.page_texts or [], chunk_size=size, overlap=overlap
                    )
                    for chunk in chunks:
                        texts.append(chunk.content)
                        meta.append(
                            {
                                "paper_id": paper.id,
                                "title": paper.title,
                                "section": chunk.section,
                                "page": chunk.page_start,
                                "content": chunk.content,
                            }
                        )
                if not texts:
                    continue
                vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
                for k in top_ks:
                    recalls: list[float] = []
                    mrrs: list[float] = []
                    ndcgs: list[float] = []
                    started = time.perf_counter()
                    for case in cases:
                        q = np.asarray(query_vectors[case["id"]], dtype=np.float32)
                        cosine = vectors @ q
                        terms = query_terms[case["id"]]
                        idf = _document_frequencies(
                            terms, [f"{m['title']} {m['content']}" for m in meta]
                        )
                        lexical = np.asarray(
                            [
                                _lexical_overlap(terms, m["content"], idf)
                                for m in meta
                            ],
                            dtype=np.float32,
                        )
                        title_hit = np.asarray(
                            [
                                _lexical_overlap(terms, m["title"], idf)
                                for m in meta
                            ],
                            dtype=np.float32,
                        )
                        if embedder.is_local:
                            scores = 0.45 * cosine + 0.55 * lexical + 0.10 * title_hit
                        else:
                            scores = 0.82 * cosine + 0.18 * lexical + 0.04 * title_hit
                        order = np.argsort(-scores)[: max_top_k,]
                        ranked: list[str] = []
                        for idx in order:
                            ranked.append(meta[int(idx)]["paper_id"])
                        m = _paper_level_metrics(
                            ranked, set(relevant.get(case["id"], [])), k
                        )
                        recalls.append(m["recall"])
                        mrrs.append(m["mrr"])
                        ndcgs.append(m["ndcg"])
                    latency = int((time.perf_counter() - started) * 1000)
                    rows.append(
                        {
                            "chunk_size": size,
                            "overlap": overlap,
                            "top_k": k,
                            "recall_at_k": _avg(recalls),
                            "mrr": _avg(mrrs),
                            "ndcg_at_k": _avg(ndcgs),
                            "chunk_count": len(texts),
                            "avg_chunks_per_paper": round(len(texts) / max(1, len(papers)), 1),
                            "retrieval_latency_ms": latency,
                        }
                    )
                logger.info("网格实验完成 chunk=%s overlap=%s", size, overlap)
                run.metrics = {
                    "progress": {
                        "chunk_size": size,
                        "overlap": overlap,
                        "completed_configs": len(rows) // max(1, len(top_ks)),
                    }
                }
                session.commit()

        rows.sort(key=lambda r: (-r["recall_at_k"], -r["mrr"], r["top_k"]))
        best = rows[0] if rows else {}
        run.metrics = {
            "best_config": best,
            "config_count": len(rows),
            "chunk_sizes": chunk_sizes,
            "overlaps": overlaps,
            "top_ks": top_ks,
            "case_count": len(cases),
            "reused_corpus_papers": len(papers),
        }
        run.details = rows
        run.case_count = len(cases)
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)

    def _run_prompt_grid(
        self, session: Session, run: EvaluationRun, config: dict[str, Any]
    ) -> None:
        cases = load_dataset(limit=config.get("limit") or 10)
        relevant = _resolve_relevant_ids(session, cases)
        top_k = int(config.get("top_k") or 6)
        rows: list[dict[str, Any]] = []
        all_details: list[dict[str, Any]] = []
        for variant_key in PROMPT_VARIANTS:
            details, metrics = self._agent_cases(
                session,
                cases,
                relevant,
                top_k=top_k,
                prompt_variant=variant_key,
            )
            label = PROMPT_VARIANTS[variant_key]["label"]
            rows.append(
                {
                    "variant": variant_key,
                    "label": label,
                    **{k: v for k, v in metrics.items()},
                }
            )
            all_details.append(
                {
                    "variant": variant_key,
                    "label": label,
                    "cases": [
                        {k: d[k] for k in ("id", "question", "success", "tool_f1", "latency_ms")}
                        for d in details
                    ],
                }
            )
            run.metrics = {"progress": {"completed_variants": len(rows)}}
            session.commit()
        rows.sort(key=lambda r: (-r["task_success_rate"], -r["tool_calling_accuracy"]))
        run.metrics = {
            "best_variant": rows[0] if rows else {},
            "variants": rows,
            "case_count": len(cases),
            "top_k": top_k,
        }
        run.details = all_details
        run.case_count = len(cases)
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    def get_run(self, session: Session, run_id: str, include_details: bool = True) -> dict[str, Any]:
        run = session.get(EvaluationRun, run_id)
        if not run:
            raise ValueError("评测任务不存在")
        return run_to_dict(run, include_details=include_details)

    def list_runs(self, session: Session, limit: int = 30) -> list[dict[str, Any]]:
        runs = session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        ).all()
        return [run_to_dict(r, include_details=False) for r in runs]


def run_to_dict(run: EvaluationRun, include_details: bool = True) -> dict[str, Any]:
    data = {
        "id": run.id,
        "name": run.name,
        "kind": run.kind,
        "status": run.status,
        "dataset": run.dataset,
        "config": run.config or {},
        "metrics": run.metrics or {},
        "error_message": run.error_message,
        "llm_provider": run.llm_provider,
        "embedding_model": run.embedding_model,
        "case_count": run.case_count,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
    data["summary"] = summarize_run(run)
    if include_details:
        data["details"] = run.details or []
    return data


def summarize_run(run: EvaluationRun) -> str:
    m = run.metrics or {}
    if run.kind == "retrieval_grid":
        best = m.get("best_config") or {}
        if best:
            return (
                f"最佳组合：chunk={best.get('chunk_size')} overlap={best.get('overlap')} "
                f"Top-K={best.get('top_k')}，Recall@K={best.get('recall_at_k')}"
            )
    if run.kind == "prompt_grid":
        best = m.get("best_variant") or {}
        if best:
            return (
                f"最佳 Prompt：{best.get('label')}，任务成功率="
                f"{best.get('task_success_rate')}，工具准确率={best.get('tool_calling_accuracy')}"
            )
    parts = []
    if "retrieval_recall_at_k" in m:
        parts.append(f"Recall@K={m['retrieval_recall_at_k']}")
    if "tool_calling_accuracy" in m:
        parts.append(f"工具准确率={m['tool_calling_accuracy']}")
    if "task_success_rate" in m:
        parts.append(f"任务成功率={m['task_success_rate']}")
    return "；".join(parts) or "暂无指标"


_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _service
    if _service is None:
        _service = EvaluationService()
    return _service
