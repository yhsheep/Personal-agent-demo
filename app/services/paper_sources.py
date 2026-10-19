"""论文来源适配：本地库、OpenAlex、arXiv、Crossref。

外部检索失败不会影响应用运行；界面会展示具体错误并允许重试。
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Paper
from app.utils.text import collapse_ws, sanitize_text, sanitize_value, truncate

logger = logging.getLogger(__name__)

USER_AGENT = "WenLan/0.1 (research literature agent; local academic use)"


class PaperSourceError(RuntimeError):
    pass


@dataclass
class PaperCandidate:
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = ""
    external_id: str = ""
    citation_count: int = 0

    @property
    def candidate_id(self) -> str:
        raw = f"{self.doi}|{self.arxiv_id}|{self.title.lower()}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_id"] = self.candidate_id
        return data


def _client(timeout: float = 20.0) -> httpx.Client:
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout, connect=10.0),
        "headers": {"User-Agent": USER_AGENT},
        "follow_redirects": True,
        "trust_env": True,
    }
    if settings.http_proxy_url:
        kwargs["proxy"] = settings.http_proxy_url
    return httpx.Client(**kwargs)


def _clean_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi


def _dedupe(candidates: list[PaperCandidate]) -> list[PaperCandidate]:
    seen: set[str] = set()
    out: list[PaperCandidate] = []
    for c in candidates:
        if not c.title:
            continue
        key = (c.doi or c.arxiv_id or c.title).lower()
        key = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", key)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


class OpenAlexSource:
    name = "openalex"

    @staticmethod
    def _abstract(inverted: dict[str, list[int]] | None) -> str:
        if not inverted:
            return ""
        positions: list[tuple[int, str]] = []
        for word, idxs in inverted.items():
            for idx in idxs:
                positions.append((idx, word))
        positions.sort()
        return collapse_ws(" ".join(w for _, w in positions))

    def search(self, query: str, limit: int = 8, year_from: int | None = None) -> list[PaperCandidate]:
        params: dict[str, Any] = {
            "search": query,
            "per-page": max(1, min(limit, 20)),
            "sort": "relevance_score:desc",
        }
        if year_from:
            params["filter"] = f"from_publication_date:{year_from}-01-01"
        if settings.openalex_mailto:
            params["mailto"] = settings.openalex_mailto
        try:
            with _client() as client:
                resp = client.get("https://api.openalex.org/works", params=params)
            if resp.status_code >= 400:
                raise PaperSourceError(f"OpenAlex HTTP {resp.status_code}")
            data = resp.json()
        except Exception as exc:
            raise PaperSourceError(f"OpenAlex 检索失败：{exc}") from exc

        out: list[PaperCandidate] = []
        for item in data.get("results", []):
            authors = [
                collapse_ws(a.get("author", {}).get("display_name", ""))
                for a in item.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]
            location = item.get("primary_location") or {}
            source_info = location.get("source") or {}
            best_oa = item.get("best_oa_location") or {}
            pdf_url = best_oa.get("pdf_url") or location.get("pdf_url") or ""
            landing = best_oa.get("landing_page_url") or location.get("landing_page_url") or ""
            doi = _clean_doi(item.get("doi") or "")
            url = f"https://doi.org/{doi}" if doi else (landing or item.get("id", ""))
            out.append(
                PaperCandidate(
                    title=collapse_ws(item.get("title") or item.get("display_name") or ""),
                    authors=authors[:20],
                    abstract=truncate(self._abstract(item.get("abstract_inverted_index")), 3000),
                    year=item.get("publication_year"),
                    venue=collapse_ws(source_info.get("display_name") or ""),
                    doi=doi,
                    url=url,
                    pdf_url=pdf_url or "",
                    source=self.name,
                    external_id=str(item.get("id") or "").rsplit("/", 1)[-1],
                    citation_count=int(item.get("cited_by_count") or 0),
                )
            )
        return _dedupe(out)


class ArxivSource:
    name = "arxiv"
    NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def search(self, query: str, limit: int = 8, year_from: int | None = None) -> list[PaperCandidate]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(limit, 20)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            with _client(timeout=25.0) as client:
                resp = client.get("https://export.arxiv.org/api/query", params=params)
            if resp.status_code >= 400:
                raise PaperSourceError(f"arXiv HTTP {resp.status_code}")
            root = ET.fromstring(resp.text)
        except Exception as exc:
            raise PaperSourceError(f"arXiv 检索失败：{exc}") from exc

        out: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", self.NS):
            arxiv_url = collapse_ws(entry.findtext("atom:id", default="", namespaces=self.NS))
            arxiv_id = arxiv_url.rsplit("/abs/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            published = entry.findtext("atom:published", default="", namespaces=self.NS)
            year = int(published[:4]) if published[:4].isdigit() else None
            if year_from and year and year < year_from:
                continue
            authors = [
                collapse_ws(a.findtext("atom:name", default="", namespaces=self.NS))
                for a in entry.findall("atom:author", self.NS)
            ]
            journal = entry.findtext("arxiv:journal_ref", default="", namespaces=self.NS)
            doi = entry.findtext("arxiv:doi", default="", namespaces=self.NS)
            out.append(
                PaperCandidate(
                    title=collapse_ws(entry.findtext("atom:title", default="", namespaces=self.NS)),
                    authors=[a for a in authors if a],
                    abstract=truncate(
                        collapse_ws(
                            entry.findtext("atom:summary", default="", namespaces=self.NS)
                        ),
                        3000,
                    ),
                    year=year,
                    venue=collapse_ws(journal) or "arXiv",
                    doi=_clean_doi(doi),
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    source=self.name,
                    external_id=arxiv_id,
                )
            )
        return _dedupe(out)


class CrossrefSource:
    name = "crossref"

    def search(self, query: str, limit: int = 8, year_from: int | None = None) -> list[PaperCandidate]:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": max(1, min(limit, 20)),
            "select": "DOI,title,author,issued,container-title,abstract,URL,link",
        }
        if year_from:
            params["filter"] = f"from-pub-date:{year_from}-01-01"
        try:
            with _client() as client:
                resp = client.get("https://api.crossref.org/works", params=params)
            if resp.status_code >= 400:
                raise PaperSourceError(f"Crossref HTTP {resp.status_code}")
            items = resp.json().get("message", {}).get("items", [])
        except Exception as exc:
            raise PaperSourceError(f"Crossref 检索失败：{exc}") from exc

        out: list[PaperCandidate] = []
        for item in items:
            title = collapse_ws((item.get("title") or [""])[0])
            authors = []
            for a in item.get("author") or []:
                name = collapse_ws(f"{a.get('given', '')} {a.get('family', '')}")
                if name:
                    authors.append(name)
            issued = (item.get("issued") or {}).get("date-parts") or [[None]]
            year = issued[0][0] if issued and issued[0] else None
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
            pdf_url = ""
            for link in item.get("link") or []:
                if "pdf" in (link.get("content-type") or "").lower():
                    pdf_url = link.get("URL") or ""
                    break
            doi = _clean_doi(item.get("DOI") or "")
            out.append(
                PaperCandidate(
                    title=title,
                    authors=authors[:20],
                    abstract=truncate(collapse_ws(abstract), 3000),
                    year=year,
                    venue=collapse_ws((item.get("container-title") or [""])[0]),
                    doi=doi,
                    url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                    pdf_url=pdf_url,
                    source=self.name,
                    external_id=doi,
                )
            )
        return _dedupe(out)


class PaperSearchService:
    def __init__(self) -> None:
        self.sources = {
            "openalex": OpenAlexSource(),
            "arxiv": ArxivSource(),
            "crossref": CrossrefSource(),
        }

    def search_local(self, session: Session, query: str, limit: int = 8) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            stmt = select(Paper).order_by(Paper.created_at.desc()).limit(limit)
        else:
            like = f"%{q}%"
            stmt = (
                select(Paper)
                .where(
                    or_(
                        Paper.title.ilike(like),
                        Paper.abstract.ilike(like),
                        Paper.venue.ilike(like),
                    )
                )
                .order_by(Paper.year.desc().nullslast(), Paper.created_at.desc())
                .limit(limit * 2)
            )
        papers = list(session.scalars(stmt).all())
        if q and len(papers) < limit:
            # 元数据没有命中时，用 RAG 语义检索按论文去重补充
            try:
                from app.services.rag import RAGService

                semantic = RAGService().search(session, q, top_k=limit * 3)
                existing = {p.id for p in papers}
                for item in semantic:
                    if item["paper_id"] in existing:
                        continue
                    paper = session.get(Paper, item["paper_id"])
                    if paper:
                        papers.append(paper)
                        existing.add(paper.id)
                    if len(papers) >= limit:
                        break
            except Exception as exc:
                logger.debug("本地语义补充检索失败：%s", exc)
        return [paper_to_dict(p, include_abstract=True) for p in papers[:limit]]

    def search_online(
        self, query: str, limit: int = 8, year_from: int | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        enabled = [s.strip() for s in settings.paper_sources.split(",") if s.strip()]
        if "local" in enabled:
            enabled.remove("local")
        results: list[PaperCandidate] = []
        errors: list[str] = []
        per_source = max(3, min(limit, 10))
        for name in enabled:
            source = self.sources.get(name)
            if not source:
                continue
            try:
                results.extend(source.search(query, per_source, year_from))
            except Exception as exc:
                errors.append(str(exc))
                logger.warning("%s 检索失败：%s", name, exc)
        deduped = _dedupe(results)
        # 优先开放获取、有 PDF、被引次数高的结果
        deduped.sort(
            key=lambda c: (bool(c.pdf_url), bool(c.abstract), c.citation_count, c.year or 0),
            reverse=True,
        )
        return [c.to_dict() for c in deduped[:limit]], errors


def paper_to_dict(paper: Paper, include_abstract: bool = True) -> dict[str, Any]:
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors or [],
        "abstract": paper.abstract if include_abstract else "",
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "source": paper.source,
        "language": paper.language,
        "keywords": paper.keywords or [],
        "page_count": paper.page_count,
        "chunk_count": paper.chunk_count,
        "parse_status": paper.parse_status,
        "error_message": paper.error_message,
        "file_size": paper.file_size,
        "has_pdf": bool(paper.pdf_path),
        "indexed_at": paper.indexed_at.isoformat() if paper.indexed_at else None,
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


def find_duplicate(session: Session, candidate: PaperCandidate) -> Paper | None:
    conditions = []
    if candidate.doi:
        conditions.append(Paper.doi == candidate.doi)
    if candidate.arxiv_id:
        conditions.append(Paper.arxiv_id == candidate.arxiv_id)
    if candidate.title:
        conditions.append(Paper.title == candidate.title)
    if not conditions:
        return None
    return session.scalars(select(Paper).where(or_(*conditions)).limit(1)).first()


def paper_from_candidate(candidate: PaperCandidate) -> Paper:
    return Paper(
        title=sanitize_text(candidate.title),
        authors=sanitize_value(candidate.authors),
        abstract=sanitize_text(candidate.abstract),
        year=candidate.year,
        venue=sanitize_text(candidate.venue),
        doi=sanitize_text(candidate.doi) or None,
        arxiv_id=sanitize_text(candidate.arxiv_id) or None,
        url=sanitize_text(candidate.url),
        pdf_url=sanitize_text(candidate.pdf_url),
        source=candidate.source or "import",
        language="en",
        keywords=[],
        parse_status="metadata",
    )
