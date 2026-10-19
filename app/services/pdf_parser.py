"""PDF 文本解析：PyMuPDF、标题/摘要抽取、页级文本。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz

from app.utils.text import collapse_ws, normalize_text, sanitize_text, truncate

logger = logging.getLogger(__name__)


class PDFParseError(RuntimeError):
    pass


@dataclass
class ParsedPDF:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    pages: list[dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.get("text", "") for p in self.pages)


def _clean_line(line: str) -> str:
    line = sanitize_text(line).replace("\u00ad", "")
    line = re.sub(r"(\w)-\n(\w)", r"\1\2", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _is_probable_page_number(text: str) -> bool:
    t = text.strip()
    return bool(re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", t))


def _extract_page_blocks(page: fitz.Page) -> list[str]:
    blocks = page.get_text("blocks", sort=True) or []
    out: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = _clean_line(str(block[4]))
        if not text or _is_probable_page_number(text):
            continue
        out.append(text)
    return out


def _remove_running_headers(pages: list[list[str]]) -> list[list[str]]:
    """移除在多数页面重复出现的页眉/页脚。"""
    if len(pages) < 4:
        return pages
    from collections import Counter

    edge_counter: Counter[str] = Counter()
    for blocks in pages:
        for line in blocks[:1] + blocks[-1:]:
            key = collapse_ws(line).lower()
            if 3 <= len(key) <= 120:
                edge_counter[key] += 1
    threshold = max(3, int(len(pages) * 0.4))
    repeated = {k for k, v in edge_counter.items() if v >= threshold}
    cleaned: list[list[str]] = []
    for blocks in pages:
        cleaned.append(
            [b for b in blocks if collapse_ws(b).lower() not in repeated]
        )
    return cleaned


def _guess_title(doc: fitz.Document, first_blocks: list[str]) -> str:
    meta_title = collapse_ws(sanitize_text(str(doc.metadata.get("title") or "")))
    if 8 <= len(meta_title) <= 300 and not meta_title.lower().endswith(".pdf"):
        return meta_title
    # 用首页最大字号的 span 推断标题
    try:
        page = doc[0]
        spans: list[tuple[float, str]] = []
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                text = sanitize_text("".join(span.get("text", "") for span in line.get("spans", [])))
                if not text.strip():
                    continue
                size = max(float(s.get("size", 0)) for s in line.get("spans", []))
                spans.append((size, text.strip()))
        if spans:
            max_size = max(size for size, _ in spans)
            title_parts = [t for size, t in spans if size >= max_size - 0.8][:3]
            title = collapse_ws(" ".join(title_parts))
            if 8 <= len(title) <= 300:
                return title
    except Exception:
        pass
    for block in first_blocks[:5]:
        if 8 <= len(block) <= 250 and not block.lower().startswith("arxiv"):
            return block
    return "未命名论文"


def _guess_authors(first_blocks: list[str], title: str) -> list[str]:
    for block in first_blocks[:8]:
        block = sanitize_text(block)
        if block == title or len(block) > 400:
            continue
        low = block.lower()
        if any(k in low for k in ("abstract", "摘要", "introduction", "keywords")):
            break
        if re.search(r"@|university|institute|department|laboratory", low):
            continue
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", block) or "，" in block or "," in block:
            cleaned = re.sub(r"[\d*†‡§¶]+", "", block)
            parts = re.split(r",|;|，|、|\band\b|&", cleaned)
            names = [
                collapse_ws(p)
                for p in parts
                if 2 <= len(collapse_ws(p)) <= 60 and not re.search(r"\d{4}", p)
            ]
            if names:
                return names[:12]
    return []


def _extract_abstract(text: str) -> str:
    text = sanitize_text(text)
    match = re.search(
        r"(?im)^\s*(?:abstract|摘\s*要)\s*[:：.\-—]?\s*(.{50,4000}?)"
        r"(?=\n\s*(?:keywords?|index terms|1\s+introduction|introduction|"
        r"关键词|引言|一、|1\.\s)|$)",
        text,
        re.S,
    )
    if match:
        abstract = collapse_ws(match.group(1))
        return truncate(abstract, 2500)
    # 没有显式标题时，取首页前 1/3 中较长段落
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if len(p) > 200]
    if paragraphs:
        return truncate(collapse_ws(paragraphs[0]), 1800)
    return ""


def parse_pdf(path: str | Path, max_pages: int = 300) -> ParsedPDF:
    path = Path(path)
    if not path.exists():
        raise PDFParseError("PDF 文件不存在")
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PDFParseError(f"无法打开 PDF：{exc}") from exc
    try:
        if doc.needs_pass:
            raise PDFParseError("PDF 已加密，请先移除密码后再上传")
        if doc.page_count == 0:
            raise PDFParseError("PDF 没有可用页面")
        if doc.page_count > max_pages:
            raise PDFParseError(f"PDF 页数 {doc.page_count} 超过限制 {max_pages}")

        raw_pages: list[list[str]] = []
        for i in range(doc.page_count):
            raw_pages.append(_extract_page_blocks(doc[i]))
        raw_pages = _remove_running_headers(raw_pages)

        pages: list[dict[str, Any]] = []
        total_chars = 0
        for i, blocks in enumerate(raw_pages, start=1):
            text = sanitize_text("\n\n".join(blocks)).strip()
            total_chars += len(text)
            pages.append({"page": i, "text": text})

        if total_chars < 120:
            raise PDFParseError("未提取到足够文本，可能是扫描版 PDF，请先进行 OCR")

        first_text = pages[0]["text"] if pages else ""
        title = _guess_title(doc, raw_pages[0] if raw_pages else [])
        authors = _guess_authors(raw_pages[0] if raw_pages else [], title)
        abstract = _extract_abstract(first_text)
        return ParsedPDF(
            title=title,
            authors=authors,
            abstract=abstract,
            pages=pages,
            page_count=doc.page_count,
            metadata={
                "producer": str(doc.metadata.get("producer") or ""),
                "creator": str(doc.metadata.get("creator") or ""),
                "subject": str(doc.metadata.get("subject") or ""),
                "total_chars": total_chars,
            },
        )
    finally:
        doc.close()


def _wrap_for_pdf(text: str, fontsize: float, max_width: float, fontname: str) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw.strip():
            lines.append("")
            continue
        current = ""
        for ch in raw:
            candidate = current + ch
            try:
                width = fitz.get_text_length(candidate, fontname=fontname, fontsize=fontsize)
            except Exception:
                width = len(candidate) * fontsize * 0.55
            if width > max_width and current:
                cut = current.rfind(" ")
                if cut > len(current) * 0.6:
                    lines.append(current[:cut].rstrip())
                    current = current[cut + 1 :] + ch
                else:
                    lines.append(current.rstrip())
                    current = ch
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
    return lines


def create_pdf_from_text(
    text: str,
    out_path: str | Path,
    title: str = "",
    authors: str = "",
) -> Path:
    """演示/测试辅助：把纯文本逐行写入 PDF（避免 textbox 溢出丢字）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    margin = 58
    width, height = fitz.paper_size("a4")
    fontname = "china-s"
    page = doc.new_page(width=width, height=height)
    y = margin + 8
    max_width = width - 2 * margin

    def new_page() -> fitz.Page:
        nonlocal y
        p = doc.new_page(width=width, height=height)
        y = margin + 8
        return p

    def write_lines(lines: list[str], size: float, color: tuple[float, float, float], leading: float) -> None:
        nonlocal page, y
        for line in lines:
            if y + leading > height - margin:
                page = new_page()
            if line:
                page.insert_text(
                    (margin, y),
                    line,
                    fontsize=size,
                    fontname=fontname,
                    color=color,
                )
            y += leading

    if title:
        title_lines = _wrap_for_pdf(title, 17, max_width, fontname)
        write_lines(title_lines[:4], 17, (0.08, 0.12, 0.18), 23)
        y += 6
    if authors:
        author_lines = _wrap_for_pdf(authors, 9.5, max_width, fontname)
        write_lines(author_lines[:4], 9.5, (0.35, 0.4, 0.45), 14)
    y += 10

    paragraphs = [p.strip() for p in normalize_text(text).split("\n") if p.strip()]
    for para in paragraphs:
        # 小标题：短且无句号
        is_heading = len(para) < 80 and not re.search(r"[。.!?；;]$", para)
        size = 12.5 if is_heading else 10.2
        color = (0.05, 0.22, 0.21) if is_heading else (0.12, 0.14, 0.18)
        lines = _wrap_for_pdf(para, size, max_width, fontname)
        if is_heading:
            y += 4
        write_lines(lines, size, color, size * 1.62)
        y += 4 if is_heading else 6

    doc.set_metadata({"title": title, "author": authors, "creator": "WenLan demo corpus"})
    doc.save(str(out_path), garbage=3, deflate=True)
    doc.close()
    return out_path
