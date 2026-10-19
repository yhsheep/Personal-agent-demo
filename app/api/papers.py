"""文献库、PDF 上传、发现与 RAG 检索 API。"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, Paper
from app.schemas import CandidateImport, RAGSearchRequest, ReindexRequest
from app.services.library import (
    LibraryError,
    LibraryService,
    paper_detail,
    resolve_pdf_path,
)
from app.services.paper_sources import PaperSearchService, paper_to_dict
from app.services.pdf_parser import PDFParseError
from app.services.rag import RAGService, citation_list
from app.services.seed import seed_demo
from app.utils.text import truncate

logger = logging.getLogger(__name__)
router = APIRouter(tags=["library"])
library_service = LibraryService()
search_service = PaperSearchService()
rag_service = RAGService()


def _get_paper(db: Session, paper_id: str) -> Paper:
    paper = db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="论文不存在")
    return paper


@router.get("/papers")
def list_papers(
    q: str = "",
    status: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(Paper)
    count_stmt = select(func.count(Paper.id))
    if q.strip():
        like = f"%{q.strip()}%"
        cond = Paper.title.ilike(like) | Paper.abstract.ilike(like) | Paper.venue.ilike(like)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    if status:
        stmt = stmt.where(Paper.parse_status == status)
        count_stmt = count_stmt.where(Paper.parse_status == status)
    total = db.scalar(count_stmt) or 0
    papers = db.scalars(
        stmt.order_by(Paper.created_at.desc()).offset(offset).limit(limit)
    ).all()
    return {
        "total": total,
        "items": [paper_to_dict(p) for p in papers],
        "limit": limit,
        "offset": offset,
    }


@router.get("/papers/{paper_id}")
def get_paper(paper_id: str, db: Session = Depends(get_db)) -> dict:
    paper = _get_paper(db, paper_id)
    data = paper_detail(db, paper)
    data["chunk_preview"] = [
        {
            "id": c.id,
            "section": c.section,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chars": c.char_count,
            "preview": truncate(c.content, 260),
        }
        for c in db.scalars(
            select(Chunk).where(Chunk.paper_id == paper.id).order_by(Chunk.chunk_index).limit(8)
        ).all()
    ]
    return data


@router.get("/papers/{paper_id}/chunks")
def get_chunks(
    paper_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    _get_paper(db, paper_id)
    total = db.scalar(
        select(func.count(Chunk.id)).where(Chunk.paper_id == paper_id)
    ) or 0
    chunks = db.scalars(
        select(Chunk)
        .where(Chunk.paper_id == paper_id)
        .order_by(Chunk.chunk_index)
        .offset(offset)
        .limit(limit)
    ).all()
    return {
        "total": total,
        "items": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "char_count": c.char_count,
                "token_count": c.token_count,
                "content": c.content,
            }
            for c in chunks
        ],
    }


@router.get("/papers/{paper_id}/pdf")
def get_pdf(paper_id: str, db: Session = Depends(get_db)):
    paper = _get_paper(db, paper_id)
    path = resolve_pdf_path(paper)
    if not path:
        raise HTTPException(status_code=404, detail="该论文没有本地 PDF")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"{paper.title[:80]}.pdf",
    )


@router.post("/papers/upload")
def upload_paper(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    authors: str = Form(default=""),
    db: Session = Depends(get_db),
) -> dict:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    temp = settings.uploads_dir / f".upload-{uuid.uuid4().hex}.pdf"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with temp.open("wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF 超过 {settings.max_upload_mb}MB 限制",
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="文件为空")
        with temp.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="文件不是有效的 PDF")
        author_list = [
            a.strip() for a in authors.replace("、", ",").replace(";", ",").split(",") if a.strip()
        ]
        paper, created = library_service.import_pdf(
            db,
            temp,
            original_name=filename,
            title=title.strip() if title else None,
            authors=author_list or None,
        )
        db.commit()
        db.refresh(paper)
        return {
            "paper": paper_to_dict(paper),
            "created": created,
            "duplicate": not created,
            "chunk_count": paper.chunk_count,
            "message": "上传并建立索引成功" if created else "相同 PDF 已存在，已返回已有论文",
        }
    except HTTPException:
        db.rollback()
        raise
    except PDFParseError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LibraryError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("上传 PDF 失败")
        raise HTTPException(status_code=500, detail=f"处理 PDF 失败：{exc}") from exc
    finally:
        temp.unlink(missing_ok=True)
        file.file.close()


@router.delete("/papers/{paper_id}")
def delete_paper(paper_id: str, db: Session = Depends(get_db)) -> dict:
    paper = _get_paper(db, paper_id)
    title = paper.title
    library_service.delete_paper(db, paper)
    db.commit()
    return {"deleted": True, "paper_id": paper_id, "title": title}


@router.post("/papers/{paper_id}/reindex")
def reindex_paper(
    paper_id: str,
    body: ReindexRequest,
    db: Session = Depends(get_db),
) -> dict:
    paper = _get_paper(db, paper_id)
    try:
        result = library_service.reindex_paper(
            db, paper, chunk_size=body.chunk_size, overlap=body.overlap
        )
        db.commit()
        return {"ok": True, **result}
    except LibraryError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/library/stats")
def library_stats(db: Session = Depends(get_db)) -> dict:
    return library_service.stats(db)


@router.post("/library/seed")
def seed_library(
    download: bool = Query(default=False, description="是否尝试下载真实 arXiv PDF"),
    db: Session = Depends(get_db),
) -> dict:
    try:
        result = seed_demo(db, download=download)
        return result
    except Exception as exc:
        db.rollback()
        logger.exception("导入示例文献库失败")
        raise HTTPException(status_code=500, detail=f"导入失败：{exc}") from exc


@router.post("/library/reindex")
def reindex_all(
    body: ReindexRequest | None = None,
    db: Session = Depends(get_db),
) -> dict:
    body = body or ReindexRequest()
    papers = db.scalars(select(Paper).where(Paper.page_texts.is_not(None))).all()
    results = []
    for paper in papers:
        try:
            result = library_service.reindex_paper(
                db, paper, chunk_size=body.chunk_size, overlap=body.overlap
            )
            db.commit()
            results.append({"paper_id": paper.id, "ok": True, **result})
        except Exception as exc:
            db.rollback()
            results.append({"paper_id": paper.id, "ok": False, "error": str(exc)})
    return {
        "total": len(papers),
        "succeeded": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "results": results,
    }


@router.post("/rag/search")
def rag_search(body: RAGSearchRequest, db: Session = Depends(get_db)) -> dict:
    chunks = rag_service.search(
        db,
        body.query,
        top_k=body.top_k,
        paper_ids=body.paper_ids,
        min_score=body.min_score,
    )
    return {
        "query": body.query,
        "count": len(chunks),
        "results": chunks,
        "citations": citation_list(chunks),
    }


@router.get("/discovery/search")
def discovery_search(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=8, ge=1, le=30),
    year_from: int | None = None,
    scope: str = Query(default="all", pattern="^(all|local|online)$"),
    db: Session = Depends(get_db),
) -> dict:
    local: list[dict] = []
    online: list[dict] = []
    errors: list[str] = []
    if scope in {"all", "local"}:
        local = search_service.search_local(db, q, limit)
        for item in local:
            item["source"] = item.get("source") or "library"
            item["in_library"] = True
    if scope in {"all", "online"}:
        online, errors = search_service.search_online(q, limit, year_from)
        for item in online:
            item["in_library"] = False
    seen = {item.get("doi") or item.get("title") for item in local}
    merged = local + [
        item for item in online if (item.get("doi") or item.get("title")) not in seen
    ]
    return {
        "query": q,
        "local_count": len(local),
        "online_count": len(online),
        "items": merged[: limit * 2],
        "errors": errors,
    }


@router.post("/discovery/import")
def import_candidate(body: CandidateImport, db: Session = Depends(get_db)) -> dict:
    candidate = body.to_candidate()
    try:
        paper, created, message = library_service.import_candidate(
            db, candidate, fetch_pdf=body.fetch_pdf
        )
        db.commit()
        db.refresh(paper)
        return {
            "paper": paper_to_dict(paper),
            "created": created,
            "message": message,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("导入论文失败")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
