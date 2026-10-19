"""文献库：上传、解析、索引、去重、删除与重新索引。"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import BASE_DIR, settings
from app.models import Chunk, Paper
from app.services.embeddings import get_embedding_service
from app.services.paper_sources import (
    PaperCandidate,
    find_duplicate,
    paper_from_candidate,
    paper_to_dict,
)
from app.services.pdf_parser import parse_pdf
from app.services.rag import RAGService
from app.utils.text import extract_keywords, sanitize_text, sanitize_value

logger = logging.getLogger(__name__)


class LibraryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return "zh" if cjk > max(20, len(text) * 0.08) else "en"


def _pdf_relative_path(paper_id: str) -> str:
    return (Path("data") / "uploads" / f"{paper_id}.pdf").as_posix()


def resolve_pdf_path(paper: Paper) -> Path | None:
    if not paper.pdf_path:
        return None
    p = Path(paper.pdf_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p if p.exists() else None


class LibraryService:
    def __init__(self) -> None:
        self.rag = RAGService()

    def import_pdf(
        self,
        session: Session,
        source_path: str | Path,
        original_name: str = "",
        *,
        title: str | None = None,
        authors: list[str] | None = None,
        paper_id: str | None = None,
        source: str = "upload",
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> tuple[Paper, bool]:
        """返回 (paper, created)。重复文件返回已有论文。"""
        path = Path(source_path)
        if not path.exists():
            raise LibraryError("上传文件不存在")
        sha = _sha256(path)
        existing = session.scalars(select(Paper).where(Paper.pdf_sha256 == sha)).first()
        if existing:
            return existing, False

        parsed = parse_pdf(path, max_pages=settings.max_pdf_pages)
        paper = Paper(
            title=sanitize_text(
                (title or parsed.title or Path(original_name).stem or "未命名论文").strip()
            ),
            authors=sanitize_value(
                authors if authors is not None else parsed.authors
            ),
            abstract=sanitize_text(parsed.abstract),
            year=None,
            venue="",
            doi=None,
            arxiv_id=None,
            url="",
            source=source,
            language=_detect_language(parsed.full_text),
            keywords=sanitize_value(
                extract_keywords(parsed.abstract + " " + parsed.title, limit=10)
            ),
            raw_text=sanitize_text(parsed.full_text),
            page_texts=sanitize_value(parsed.pages),
            pdf_sha256=sha,
            file_size=path.stat().st_size,
            page_count=parsed.page_count,
            parse_status="parsed",
            error_message="",
        )
        if metadata:
            for key in ("year", "venue", "doi", "arxiv_id", "url", "pdf_url", "source"):
                if metadata.get(key):
                    setattr(paper, key, sanitize_value(metadata[key]))
            if metadata.get("abstract") and not paper.abstract:
                paper.abstract = sanitize_text(str(metadata["abstract"]))
            if metadata.get("authors") and authors is None:
                paper.authors = sanitize_value(metadata["authors"])
            if metadata.get("keywords"):
                paper.keywords = sanitize_value(list(dict.fromkeys(
                    list(metadata["keywords"]) + list(paper.keywords or [])
                )))
        session.add(paper)
        session.flush()

        target = settings.uploads_dir / f"{paper.id}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        paper.pdf_path = _pdf_relative_path(paper.id)

        try:
            self.rag.index_paper(
                session,
                paper,
                parsed.pages,
                chunk_size=chunk_size,
                overlap=overlap,
                replace=True,
            )
        except Exception as exc:
            paper.parse_status = "parse_failed"
            paper.error_message = f"向量化失败：{exc}"
            target.unlink(missing_ok=True)
            raise LibraryError(f"PDF 解析成功，但向量化失败：{exc}") from exc
        return paper, True

    def import_candidate(
        self,
        session: Session,
        candidate: PaperCandidate,
        *,
        fetch_pdf: bool = True,
    ) -> tuple[Paper, bool, str]:
        """导入检索结果。返回 (paper, created, message)。"""
        existing = find_duplicate(session, candidate)
        if existing:
            if fetch_pdf and not existing.pdf_path and existing.arxiv_id:
                try:
                    self._fetch_and_index_arxiv(session, existing)
                    return existing, False, "已存在，已补充下载 PDF 并建立索引"
                except Exception as exc:
                    return existing, False, f"论文已存在；PDF 自动下载失败：{exc}"
            return existing, False, "论文已存在"

        paper = paper_from_candidate(candidate)
        session.add(paper)
        session.flush()
        if fetch_pdf and candidate.arxiv_id:
            try:
                self._fetch_and_index_arxiv(session, paper, candidate)
                return paper, True, "已导入并建立向量索引"
            except Exception as exc:
                logger.warning("自动下载 arXiv PDF 失败：%s", exc)
                paper.parse_status = "metadata"
                paper.error_message = f"仅导入元数据；PDF 下载失败：{exc}"
                return paper, True, "已导入元数据（PDF 自动下载失败，可手动上传）"
        return paper, True, "已导入元数据（上传 PDF 后可进行 RAG 问答）"

    def _fetch_and_index_arxiv(
        self,
        session: Session,
        paper: Paper,
        candidate: PaperCandidate | None = None,
    ) -> None:
        arxiv_id = (candidate.arxiv_id if candidate else paper.arxiv_id) or ""
        if not arxiv_id:
            raise LibraryError("缺少 arXiv ID")
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        tmp_path = settings.uploads_dir / f".download-{paper.id}.pdf"
        max_bytes = settings.max_upload_mb * 1024 * 1024
        try:
            with httpx.Client(
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "WenLan/0.1 academic-use"},
            ) as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise LibraryError(f"下载 PDF 失败：HTTP {resp.status_code}")
                    with tmp_path.open("wb") as fh:
                        size = 0
                        for chunk in resp.iter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise LibraryError("PDF 超过大小限制")
                            fh.write(chunk)
            with tmp_path.open("rb") as fh:
                if fh.read(5) != b"%PDF-":
                    raise LibraryError("下载内容不是 PDF")
            parsed = parse_pdf(tmp_path, max_pages=settings.max_pdf_pages)
            paper.raw_text = parsed.full_text
            paper.page_texts = parsed.pages
            paper.page_count = parsed.page_count
            paper.file_size = tmp_path.stat().st_size
            if not paper.abstract and parsed.abstract:
                paper.abstract = parsed.abstract
            if not paper.authors and parsed.authors:
                paper.authors = parsed.authors
            paper.pdf_sha256 = _sha256(tmp_path)
            paper.pdf_path = _pdf_relative_path(paper.id)
            target = settings.uploads_dir / f"{paper.id}.pdf"
            shutil.move(str(tmp_path), target)
            self.rag.index_paper(session, paper, parsed.pages, replace=True)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def reindex_paper(
        self,
        session: Session,
        paper: Paper,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        if not paper.page_texts:
            raise LibraryError("这篇论文没有可重建索引的原文，请重新上传 PDF")
        result = self.rag.index_paper(
            session,
            paper,
            paper.page_texts,
            chunk_size=chunk_size,
            overlap=overlap,
            replace=True,
        )
        return {
            "paper_id": result.paper_id,
            "chunk_count": result.chunk_count,
            "embedding_model": result.embedding_model,
            "elapsed_ms": result.elapsed_ms,
        }

    def delete_paper(self, session: Session, paper: Paper) -> None:
        pdf = resolve_pdf_path(paper)
        session.delete(paper)
        session.flush()
        if pdf and pdf.exists():
            pdf.unlink(missing_ok=True)

    def stats(self, session: Session) -> dict[str, Any]:
        from sqlalchemy import func

        paper_count = session.scalar(select(func.count(Paper.id))) or 0
        chunk_count = session.scalar(select(func.count(Chunk.id))) or 0
        indexed = (
            session.scalar(
                select(func.count(Paper.id)).where(Paper.parse_status == "indexed")
            )
            or 0
        )
        pages = session.scalar(select(func.sum(Paper.page_count))) or 0
        return {
            "paper_count": paper_count,
            "chunk_count": chunk_count,
            "indexed_papers": indexed,
            "page_count": pages,
            "embedding": get_embedding_service().status(),
        }


def index_status(paper: Paper) -> str:
    if paper.parse_status == "indexed":
        return "已索引"
    if paper.parse_status == "parse_failed":
        return "解析失败"
    return "仅元数据"


def paper_detail(session: Session, paper: Paper) -> dict[str, Any]:
    data = paper_to_dict(paper, include_abstract=True)
    data["raw_text_preview"] = (paper.raw_text or "")[:1200]
    data["sections"] = sorted(
        {
            c.section
            for c in session.scalars(
                select(Chunk).where(Chunk.paper_id == paper.id).limit(50)
            ).all()
            if c.section
        }
    )
    data["updated_at"] = paper.updated_at.isoformat() if paper.updated_at else None
    return data
