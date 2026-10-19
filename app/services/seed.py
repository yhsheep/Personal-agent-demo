"""示例文献库：真实 arXiv 论文 + 离线笔记 PDF 回退。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import BASE_DIR, settings
from app.models import Paper
from app.services.library import LibraryService
from app.services.paper_sources import PaperCandidate
from app.services.pdf_parser import create_pdf_from_text
from app.services.seed_aliases import SEED_ALIASES

logger = logging.getLogger(__name__)

SEED_JSON = BASE_DIR / "data" / "seed" / "papers.json"
SEED_PDF_DIR = BASE_DIR / "data" / "seed" / "pdfs"


def load_seed_entries() -> list[dict[str, Any]]:
    return json.loads(SEED_JSON.read_text(encoding="utf-8"))


def _fallback_text(entry: dict[str, Any]) -> str:
    notes = str(entry.get("demo_notes", "")).strip()
    sentences = [s.strip() for s in notes.replace("\n", " ").split(". ") if s.strip()]
    midpoint = max(1, len(sentences) // 2)
    method_notes = ". ".join(sentences[:midpoint]).strip()
    result_notes = ". ".join(sentences[midpoint:]).strip()
    if method_notes and not method_notes.endswith("."):
        method_notes += "."
    if result_notes and not result_notes.endswith("."):
        result_notes += "."
    return (
        f"Abstract\n{entry.get('abstract', '')}\n\n"
        f"1 Introduction\n{entry.get('abstract', '')}\n\n"
        f"2 Method and Contributions\n{method_notes}\n\n"
        f"3 Experiments and Findings\n{result_notes}"
    )


def _download_pdf(entry: dict[str, Any], target: Path, timeout: float = 60.0) -> bool:
    arxiv_id = entry.get("arxiv_id")
    if not arxiv_id:
        return False
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "WenLan/0.1 (academic demo corpus)"},
        ) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    logger.warning("下载 %s 失败：HTTP %s", arxiv_id, resp.status_code)
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        if target.read_bytes()[:5] != b"%PDF-":
            target.unlink(missing_ok=True)
            return False
        return True
    except Exception as exc:
        logger.warning("下载 %s 异常：%s", arxiv_id, exc)
        target.unlink(missing_ok=True)
        return False


def seed_demo(
    session: Session,
    *,
    download: bool = True,
    force: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    entries = load_seed_entries()
    library = LibraryService()
    created = 0
    skipped = 0
    failed = 0
    details: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries, start=1):
        title = entry["title"]
        existing = session.scalars(
            select(Paper).where(
                (Paper.arxiv_id == entry.get("arxiv_id"))
                | (Paper.title == title)
            )
        ).first()
        if existing:
            skipped += 1
            details.append({"title": title, "status": "skipped", "paper_id": existing.id})
            if progress:
                progress(
                    {
                        "index": idx,
                        "total": len(entries),
                        "title": title,
                        "status": "skipped",
                    }
                )
            continue

        key = entry.get("key") or entry.get("arxiv_id") or f"paper-{idx}"
        pdf_path = SEED_PDF_DIR / f"{key}.pdf"
        source_note = "arxiv-pdf"
        if not pdf_path.exists() and download:
            ok = _download_pdf(entry, pdf_path)
            if not ok:
                source_note = "demo-note"
        elif not pdf_path.exists():
            source_note = "demo-note"

        temp_created = False
        if not pdf_path.exists():
            pdf_path = settings.data_dir / "seed" / f"{key}-notes.pdf"
            create_pdf_from_text(
                _fallback_text(entry),
                pdf_path,
                title=title,
                authors=", ".join(entry.get("authors", [])[:6]),
            )
            temp_created = True

        candidate = PaperCandidate(
            title=title,
            authors=entry.get("authors", []),
            abstract=entry.get("abstract", ""),
            year=entry.get("year"),
            venue=entry.get("venue", ""),
            doi=entry.get("doi", ""),
            arxiv_id=entry.get("arxiv_id", ""),
            url=entry.get("url", ""),
            pdf_url=entry.get("pdf_url", ""),
            source="seed-arxiv" if source_note == "arxiv-pdf" else "seed-demo",
        )
        try:
            paper, was_created = library.import_pdf(
                session,
                pdf_path,
                original_name=f"{key}.pdf",
                title=title,
                authors=entry.get("authors", []),
                source=candidate.source,
                metadata={
                    "year": candidate.year,
                    "venue": candidate.venue,
                    "doi": candidate.doi,
                    "arxiv_id": candidate.arxiv_id,
                    "url": candidate.url,
                    "pdf_url": candidate.pdf_url,
                    "abstract": candidate.abstract,
                    "keywords": SEED_ALIASES.get(key, []),
                },
            )
            session.commit()
            if was_created:
                created += 1
            else:
                skipped += 1
            details.append(
                {
                    "title": title,
                    "status": "created" if was_created else "skipped",
                    "paper_id": paper.id,
                    "chunks": paper.chunk_count,
                    "source": source_note,
                }
            )
            if progress:
                progress(
                    {
                        "index": idx,
                        "total": len(entries),
                        "title": title,
                        "status": "created" if was_created else "skipped",
                        "chunks": paper.chunk_count,
                    }
                )
        except Exception as exc:
            session.rollback()
            logger.exception("导入示例论文失败：%s", title)
            failed += 1
            details.append({"title": title, "status": "failed", "error": str(exc)})
            if progress:
                progress(
                    {
                        "index": idx,
                        "total": len(entries),
                        "title": title,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
        finally:
            if temp_created and pdf_path.exists():
                pdf_path.unlink(missing_ok=True)

    return {
        "total": len(entries),
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }
