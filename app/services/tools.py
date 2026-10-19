"""Agent 工具定义与执行。工具结果同时用于生成回答和界面展示。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Paper
from app.services.library import paper_detail
from app.services.paper_sources import PaperSearchService
from app.services.rag import RAGService
from app.utils.text import truncate

logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int, low: int = 1, high: int = 20) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


@dataclass
class ToolContext:
    session: Session
    rag: RAGService
    search: PaperSearchService
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_by_chunk: dict[str, int] = field(default_factory=dict)
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    default_top_k: int = 6

    def add_citations(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id in self.citation_by_chunk:
                index = self.citation_by_chunk[chunk_id]
            else:
                index = len(self.citations) + 1
                if chunk_id:
                    self.citation_by_chunk[chunk_id] = index
                self.citations.append(
                    {
                        "index": index,
                        "chunk_id": chunk_id,
                        "paper_id": chunk.get("paper_id"),
                        "paper_title": chunk.get("paper_title"),
                        "authors": chunk.get("authors") or [],
                        "year": chunk.get("year"),
                        "venue": chunk.get("venue") or "",
                        "section": chunk.get("section") or "",
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "score": round(float(chunk.get("score") or 0.0), 4),
                        "snippet": truncate(str(chunk.get("content") or ""), 480),
                    }
                )
            item = dict(chunk)
            item["citation_index"] = index
            enriched.append(item)
        return enriched


class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": "检索文献库中的论文元数据。适合找论文、推荐论文、按主题浏览。"
                    "当本地没有结果且允许在线检索时，会自动补充在线结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索关键词或研究主题"},
                            "limit": {
                                "type": "integer",
                                "description": "返回数量，1-10，默认 5",
                            },
                            "year_from": {
                                "type": "integer",
                                "description": "最早发表年份，可省略",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_paper_detail",
                    "description": "查询单篇论文的详细信息：作者、年份、摘要、DOI、章节、索引状态。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "论文 ID（32 位十六进制），如果已知则优先提供",
                            },
                            "title": {
                                "type": "string",
                                "description": "论文标题或标题片段，用于模糊定位",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_library",
                    "description": "在本地论文 PDF 知识库中做语义检索，返回带页码的原文片段。"
                    "回答方法、实验、结论、数字等问题前必须调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "问题或检索语句"},
                            "top_k": {
                                "type": "integer",
                                "description": "返回片段数，1-8，默认 6",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_library_papers",
                    "description": "列出文献库中已有的论文，适合回答“库里有哪几篇论文”。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "数量，默认 10"}
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_online_papers",
                    "description": "在 OpenAlex / arXiv 检索最新论文。仅在用户明确要求联网找论文时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "英文检索词效果更好"},
                            "limit": {"type": "integer", "description": "返回数量，默认 5"},
                            "year_from": {"type": "integer", "description": "最早年份"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        func = getattr(self, f"_tool_{name}", None)
        if not func:
            return {"error": f"未知工具：{name}"}
        if self.ctx.on_event:
            self.ctx.on_event(
                "tool_start",
                {"name": name, "arguments": arguments},
            )
        started = time.perf_counter()
        try:
            result = func(arguments or {})
            result.setdefault("ok", True)
            status = "success"
        except Exception as exc:
            logger.exception("工具 %s 执行失败", name)
            result = {"ok": False, "error": str(exc)}
            status = "error"
        elapsed = int((time.perf_counter() - started) * 1000)
        result["latency_ms"] = elapsed
        if self.ctx.on_event:
            self.ctx.on_event(
                "tool_result",
                {
                    "name": name,
                    "status": status,
                    "latency_ms": elapsed,
                    "summary": self._summary(name, result),
                },
            )
        return result

    @staticmethod
    def _summary(name: str, result: dict[str, Any]) -> str:
        if result.get("error"):
            return str(result["error"])[:200]
        if name == "search_library":
            return f"检索到 {result.get('count', 0)} 个原文片段"
        if name in {"search_papers", "search_online_papers"}:
            return f"找到 {result.get('count', 0)} 篇论文"
        if name == "get_paper_detail":
            paper = result.get("paper") or {}
            return f"已获取《{paper.get('title', '')[:60]}》详情"
        if name == "list_library_papers":
            return f"文献库共 {result.get('count', 0)} 篇论文"
        return "执行完成"

    def _tool_search_papers(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _as_str(args.get("query"))
        if not query:
            return {"error": "query 不能为空"}
        limit = _as_int(args.get("limit"), 5, 1, 10)
        year_from = args.get("year_from")
        try:
            year_from = int(year_from) if year_from else None
        except (TypeError, ValueError):
            year_from = None
        results = self.ctx.search.search_local(self.ctx.session, query, limit)
        if year_from:
            results = [r for r in results if not r.get("year") or r["year"] >= year_from]
        used_source = "library"
        if not results and settings.agent_online_search:
            online, errors = self.ctx.search.search_online(query, limit, year_from)
            for item in online:
                item["candidate_id"] = item.get("candidate_id")
            results = online
            used_source = "online"
        compact = []
        for r in results[:limit]:
            compact.append(
                {
                    "paper_id": r.get("id") or r.get("paper_id"),
                    "title": r.get("title"),
                    "authors": (r.get("authors") or [])[:6],
                    "year": r.get("year"),
                    "venue": r.get("venue"),
                    "abstract": truncate(r.get("abstract") or "", 700),
                    "has_pdf": bool(r.get("has_pdf") or r.get("chunk_count")),
                    "parse_status": r.get("parse_status", "metadata"),
                    "source": r.get("source"),
                    "doi": r.get("doi"),
                }
            )
        return {"query": query, "count": len(compact), "source": used_source, "results": compact}

    def _tool_get_paper_detail(self, args: dict[str, Any]) -> dict[str, Any]:
        paper_id = _as_str(args.get("paper_id"))
        title = _as_str(args.get("title"))
        paper: Paper | None = None
        if paper_id:
            paper = self.ctx.session.get(Paper, paper_id)
        if not paper and title:
            results = self.ctx.search.search_local(self.ctx.session, title, 3)
            if results:
                paper = self.ctx.session.get(Paper, results[0]["id"])
        if not paper:
            return {"error": "没有找到对应论文，请先使用 search_papers 检索并确认 paper_id"}
        detail = paper_detail(self.ctx.session, paper)
        detail["abstract"] = truncate(detail.get("abstract") or "", 1200)
        return {"paper": detail}

    def _tool_search_library(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _as_str(args.get("query"))
        if not query:
            return {"error": "query 不能为空"}
        top_k = _as_int(args.get("top_k"), self.ctx.default_top_k or settings.top_k, 1, 8)
        chunks = self.ctx.rag.search(self.ctx.session, query, top_k=top_k)
        enriched = self.ctx.add_citations(chunks)
        results = []
        for item in enriched:
            results.append(
                {
                    "citation_index": item["citation_index"],
                    "paper_id": item["paper_id"],
                    "paper_title": item["paper_title"],
                    "authors": (item.get("authors") or [])[:5],
                    "year": item.get("year"),
                    "section": item.get("section"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": item.get("score"),
                    "content": truncate(item.get("content") or "", 1100),
                }
            )
        return {"query": query, "count": len(results), "results": results}

    def _tool_list_library_papers(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = _as_int(args.get("limit"), 10, 1, 30)
        papers = list(
            self.ctx.session.scalars(
                select(Paper).order_by(Paper.created_at.desc()).limit(limit)
            ).all()
        )
        return {
            "count": len(papers),
            "results": [
                {
                    "paper_id": p.id,
                    "title": p.title,
                    "authors": (p.authors or [])[:5],
                    "year": p.year,
                    "chunk_count": p.chunk_count,
                    "parse_status": p.parse_status,
                }
                for p in papers
            ],
        }

    def _tool_search_online_papers(self, args: dict[str, Any]) -> dict[str, Any]:
        if not settings.agent_online_search:
            return {
                "error": "在线检索未启用。管理员可在 .env 设置 AGENT_ONLINE_SEARCH=true。",
                "results": [],
                "count": 0,
            }
        query = _as_str(args.get("query"))
        if not query:
            return {"error": "query 不能为空"}
        limit = _as_int(args.get("limit"), 5, 1, 10)
        year_from = args.get("year_from")
        try:
            year_from = int(year_from) if year_from else None
        except (TypeError, ValueError):
            year_from = None
        results, errors = self.ctx.search.search_online(query, limit, year_from)
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "candidate_id": r.get("candidate_id"),
                    "title": r.get("title"),
                    "authors": (r.get("authors") or [])[:6],
                    "year": r.get("year"),
                    "venue": r.get("venue"),
                    "abstract": truncate(r.get("abstract") or "", 600),
                    "doi": r.get("doi"),
                    "arxiv_id": r.get("arxiv_id"),
                    "url": r.get("url"),
                    "pdf_url": r.get("pdf_url"),
                    "source": r.get("source"),
                }
                for r in results
            ],
            "errors": errors,
        }


def tool_schemas() -> list[dict[str, Any]]:
    """无需数据库连接即可获取 schema（用于文档/测试）。"""
    dummy = ToolRegistry(
        ToolContext(session=None, rag=None, search=PaperSearchService())  # type: ignore[arg-type]
    )
    return dummy.schemas()


def compact_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """从工具结果中提取给界面展示的摘要，避免把全文写入会话。"""
    if name == "search_library":
        return {
            "count": result.get("count", 0),
            "citations": [
                r.get("citation_index") for r in result.get("results", [])
            ],
        }
    if name in {"search_papers", "search_online_papers", "list_library_papers"}:
        return {
            "count": result.get("count", 0),
            "titles": [r.get("title") for r in result.get("results", [])[:5]],
        }
    if name == "get_paper_detail":
        paper = result.get("paper") or {}
        return {"paper_id": paper.get("id"), "title": paper.get("title")}
    return result
