"""章节感知的滑动窗口分块。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.utils.text import estimate_tokens, normalize_text, sentence_split

_HEADING_PATTERNS = [
    re.compile(
        r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\s+)?"
        r"(?:introduction|related work|background|method(?:ology)?|approach|"
        r"experiments?|evaluation|results?|discussion|conclusions?|future work|"
        r"limitations?|references|acknowledg(?:e)?ments?|appendix|"
        r"abstract|keywords?|摘要|关键词|引言|绪论|相关工作|研究背景|方法|"
        r"实验|结果|讨论|结论|参考文献|致谢|附录)\s*$",
        re.I,
    ),
    re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,2}\s+[A-Z][A-Za-z0-9 ,:\-]{2,90}$"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*[^\n]{2,50}$"),
    re.compile(r"^第[一二三四五六七八九十\d]+[章节]\s*[^\n]{0,60}$"),
]


@dataclass
class TextChunk:
    content: str
    section: str
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int
    char_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


def _is_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 120 or t.count("\n") > 1:
        return False
    if any(p.match(t) for p in _HEADING_PATTERNS):
        return True
    return False


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:120]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    buf = ""
    for piece in re.split(r"(?<=[，,；;:：、])\s*", sentence):
        if len(buf) + len(piece) > max_chars and buf:
            parts.append(buf.strip())
            buf = piece
        else:
            buf += piece
    if buf.strip():
        parts.append(buf.strip())
    # 仍然过长则硬切
    out: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            out.extend(p[i : i + max_chars] for i in range(0, len(p), max_chars))
    return out


def _collect_units(
    pages: list[dict[str, Any]], max_sentence_chars: int
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    section = "正文"
    for page in pages:
        page_no = int(page.get("page", 1))
        text = normalize_text(str(page.get("text", "")))
        if not text:
            continue
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        for block in blocks:
            if _is_heading(block):
                section = _clean_heading(block)
                units.append(
                    {
                        "text": section,
                        "page": page_no,
                        "section": section,
                        "heading": True,
                    }
                )
                continue
            for sent in sentence_split(block):
                for piece in _split_long_sentence(sent, max_sentence_chars):
                    if len(piece.strip()) < 2:
                        continue
                    units.append(
                        {
                            "text": piece.strip(),
                            "page": page_no,
                            "section": section,
                            "heading": False,
                        }
                    )
    return units


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 600,
    overlap: int = 100,
    min_chunk_chars: int = 80,
) -> list[TextChunk]:
    """按句子边界累积，切分点优先落在章节边界。"""
    if chunk_size < 100:
        raise ValueError("chunk_size 至少为 100")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    units = _collect_units(pages, max_sentence_chars=max(120, chunk_size // 2))
    if not units:
        return []

    chunks: list[TextChunk] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    def flush(force: bool = False) -> None:
        nonlocal current, current_len
        if not current:
            return
        content = " ".join(u["text"] for u in current).strip()
        if len(content) < min_chunk_chars and not force:
            return
        chunks.append(
            TextChunk(
                content=content,
                section=current[0].get("section") or "正文",
                page_start=min(u["page"] for u in current),
                page_end=max(u["page"] for u in current),
                chunk_index=len(chunks),
                token_count=estimate_tokens(content),
                char_count=len(content),
            )
        )
        if overlap > 0:
            tail: list[dict[str, Any]] = []
            tail_len = 0
            for unit in reversed(current):
                add = len(unit["text"]) + 1
                if tail and tail_len + add > overlap:
                    break
                tail.insert(0, unit)
                tail_len += add
            current = tail if tail_len < len(content) else []
            current_len = sum(len(u["text"]) + 1 for u in current)
        else:
            current = []
            current_len = 0

    for unit in units:
        text_len = len(unit["text"]) + 1
        section_changed = current and unit["section"] != current[-1]["section"]
        if section_changed and current_len >= max(min_chunk_chars, chunk_size // 3):
            flush(force=True)
            if current and current[-1].get("section") != unit["section"]:
                current = []
                current_len = 0
        if current and current_len + text_len > chunk_size:
            flush(force=True)
        current.append(unit)
        current_len += text_len

    flush(force=True)
    # 过滤空块与完全重复块，重新编号
    seen: set[str] = set()
    cleaned: list[TextChunk] = []
    for chunk in chunks:
        key = chunk.content.strip()
        if len(key) < 30 or key in seen:
            continue
        seen.add(key)
        chunk.chunk_index = len(cleaned)
        cleaned.append(chunk)
    return cleaned
