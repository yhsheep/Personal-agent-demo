"""文本处理工具。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-.]{1,}", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z0-9])")
# PostgreSQL text/JSON 不接受 NUL；其余 C0 控制字符对学术正文没有意义。
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NON_CHAR_RE = re.compile(r"[\ufffe\uffff]")


def sanitize_text(value: str) -> str:
    """移除 PostgreSQL 无法存储的 NUL，并清理其他控制字符。

    保留换行、制表符与回车，避免破坏 PDF 文本结构。
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\x00", "")
    value = _CONTROL_RE.sub(" ", value)
    return _NON_CHAR_RE.sub("", value)


def sanitize_value(value):
    """递归清理字符串 / JSON 结构，供数据库写入前兜底使用。"""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_value(item) for item in value)
    if isinstance(value, dict):
        return {
            (sanitize_text(key) if isinstance(key, str) else key): sanitize_value(item)
            for key, item in value.items()
        }
    return value


def normalize_text(text: str) -> str:
    text = sanitize_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", sanitize_text(text or "")).strip()


def estimate_tokens(text: str) -> int:
    """不依赖 tokenizer 的保守估算，中文按字、英文按词。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    other = max(0, len(text) - cjk - sum(len(w) for w in _WORD_RE.findall(text)))
    return int(cjk * 1.0 + words * 1.35 + other * 0.25) + 1


def truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def sentence_split(text: str) -> list[str]:
    text = normalize_text(text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts or ([text] if text else [])


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    """轻量关键词抽取，用于界面展示与元数据检索。"""
    text = normalize_text(text).lower()
    tokens = [w.lower() for w in _WORD_RE.findall(text) if len(w) > 2]
    tokens += [m for m in _CJK_RE.findall(text) if 2 <= len(m) <= 8]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "into",
        "using",
        "based",
        "paper",
        "study",
        "method",
        "results",
        "我们",
        "本文",
        "研究",
        "方法",
        "结果",
        "论文",
    }
    counts = Counter(t for t in tokens if t not in stop)
    return [w for w, _ in counts.most_common(limit)]


_GLOSSARY: list[tuple[tuple[str, ...], str]] = [
    (("检索器",), "retriever encoder"),
    (("生成器",), "generator decoder"),
    (("模型",), "model"),
    (("原始", "经典", "最初"), "original classic"),
    (("检索", "召回", "查找", "搜索"), "retrieval search recall"),
    (("生成形式", "生成方式"), "sequence token formulation"),
    (("生成", "回答"), "generation generate answer"),
    (("向量", "嵌入"), "vector embedding dense"),
    (("分块", "切块", "片段"), "chunk chunking passage"),
    (("评估", "评测", "指标"), "evaluation benchmark metric"),
    (("幻觉", "事实性"), "hallucination factual"),
    (("工具调用", "函数调用"), "tool calling function"),
    (("智能体", "代理"), "agent"),
    (("提示词", "提示"), "prompt"),
    (("上下文",), "context"),
    (("知识库",), "knowledge base"),
    (("论文", "文献"), "paper document"),
    (("作者",), "author"),
    (("实验",), "experiment"),
    (("结论", "贡献", "创新"), "conclusion contribution"),
    (("方法", "架构", "框架"), "method architecture framework"),
    (("数据集", "语料"), "dataset corpus"),
    (("训练",), "training train"),
    (("推理",), "inference reasoning"),
    (("多轮", "对话"), "multi-turn dialogue conversation"),
    (("引用", "引文"), "citation cite"),
    (("重排",), "rerank"),
    (("语义",), "semantic"),
    (("准确率", "精确率"), "accuracy precision"),
    (("反思", "自我反思"), "reflection critique"),
    (("相关性",), "relevance"),
    (("支持度", "支持"), "support supported"),
    (("有用性",), "usefulness"),
    (("损失", "优化"), "loss optimization"),
    (("基线", "对比"), "baseline comparison"),
    (("长文本", "长上下文"), "long context"),
    (("摘要", "总结"), "abstract summary"),
    (("区别", "差异", "不同", "对比"), "difference unlike contrast"),
    (("未来", "方向", "后续"), "future direction"),
    (("局限", "限制", "不足"), "limitation limited"),
    (("过滤", "筛选", "保留"), "filter keep retain"),
    (("固定",), "fixed"),
    (("算术", "数学"), "arithmetic math"),
    (("常识",), "commonsense"),
    (("符号",), "symbolic"),
    (("社区",), "community"),
    (("自适应", "按需"), "adaptive on-demand"),
    (("动作", "决策"), "action correct incorrect ambiguous"),
    (("步骤", "流程"), "step process"),
    (("外部",), "external"),
    (("工具", "接口"), "tool api"),
    (("负样本",), "negative negative-sample"),
    (("批内",), "in-batch"),
    (("正样本",), "positive"),
    (("双塔", "双编码器"), "dual encoder"),
    (("token 级", "词元级", "token级"), "token-level"),
    (("类型",), "type category"),
    (("端到端",), "end-to-end"),
    (("联合",), "joint"),
    (("形式", "方式"), "form formulation"),
    (("PDF", "文档解析"), "pdf parsing"),
]


def expand_concepts(text: str) -> list[str]:
    """中英术语桥接：改善中文问题检索英文论文的效果。"""
    lower = (text or "").lower()
    out: list[str] = []
    for triggers, expansion in _GLOSSARY:
        if any(t.lower() in lower for t in triggers):
            out.extend(expansion.split())
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", value)[:80]
