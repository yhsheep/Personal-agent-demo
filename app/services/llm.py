"""LLM 客户端：OpenAI 兼容 Tool Calling + 可离线评测的 Mock 策略。"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.providers import LLMConfig, resolve_llm_config
from app.utils.text import collapse_ws, truncate

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    calls.append(tc.as_openai())
                elif isinstance(tc, dict) and "function" in tc:
                    calls.append(tc)
                elif isinstance(tc, dict):
                    calls.append(
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(
                                    tc.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            out.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": calls,
                }
            )
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", ""),
                }
            )
        else:
            out.append({"role": role, "content": msg.get("content", "")})
    return out


class OpenAICompatibleLLM:
    def __init__(self, config: LLMConfig, timeout: float | None = None) -> None:
        self.config = config
        self.timeout = timeout or settings.llm_timeout

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self.timeout, connect=20.0),
            "trust_env": True,
        }
        if settings.http_proxy_url:
            kwargs["proxy"] = settings.http_proxy_url
        return httpx.Client(**kwargs)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": _to_openai_messages(messages),
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._client() as client:
                    resp = client.post(
                        url,
                        headers=self._headers(),
                        json=self._payload(messages, tools, temperature, max_tokens, False),
                    )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    detail = resp.text[:800]
                    if resp.status_code in (401, 403):
                        detail = "API Key 无效或无权限，请检查 .env 中的配置。" + detail
                    raise LLMError(f"LLM API {resp.status_code}: {detail}")
                data = resp.json()
                return self._parse_result(data, started)
            except LLMError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM 调用失败：{last_error}")

    def _parse_result(self, data: dict[str, Any], started: float) -> LLMResult:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"LLM 返回格式异常：{str(data)[:500]}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
            tool_calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{len(tool_calls)}",
                    name=fn.get("name") or "",
                    arguments=args if isinstance(args, dict) else {"value": args},
                )
            )
        usage = data.get("usage") or {}
        return LLMResult(
            content=str(content).strip(),
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            model=self.config.model,
            provider=self.config.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = self._payload(messages, None, temperature, max_tokens, True)
        try:
            with self._client() as client:
                with client.stream(
                    "POST", url, headers=self._headers(), json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", errors="ignore")[:800]
                        raise LLMError(f"LLM stream {resp.status_code}: {body}")
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            yield str(text)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM 流式调用失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 离线 Mock 策略
# ---------------------------------------------------------------------------

_CONTENT_HINTS = (
    "方法",
    "实验",
    "结论",
    "贡献",
    "创新",
    "模型",
    "数据集",
    "指标",
    "为什么",
    "如何",
    "怎么",
    "对比",
    "区别",
    "优缺点",
    "讲了什么",
    "主要发现",
    "分析",
    "总结",
    "局限",
    "动作",
    "步骤",
    "流程",
    "机制",
    "原理",
    "作用",
    "哪几",
    "哪三种",
    "哪些类型",
    "limitation",
    "method",
    "experiment",
    "result",
)
_SEARCH_HINTS = (
    "找",
    "检索",
    "搜索",
    "推荐",
    "有哪些",
    "相关论文",
    "最新",
    "文献",
    "论文列表",
)
_GREETING_HINTS = ("你好", "hello", "hi", "你是谁", "能做什么", "帮助", "怎么用")


def _last_user(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                payload = json.loads(msg.get("content") or "{}")
            except json.JSONDecodeError:
                payload = {"raw": msg.get("content")}
            results.append(
                {
                    "name": msg.get("name", ""),
                    "call_id": msg.get("tool_call_id", ""),
                    "payload": payload,
                }
            )
    return results


def _clean_query(query: str) -> str:
    q = collapse_ws(query)
    q = re.sub(
        r"^(请|麻烦|帮我|帮忙|我想|想请你|能不能|可以)?\s*(找|检索|搜索|推荐|查|列出)"
        r"(一下|一些|几个|下)?\s*(关于|有关|与)?\s*",
        "",
        q,
    )
    q = re.sub(r"(的)?(相关)?(论文|文献|资料|研究)(有哪些|有什么)?[？?]?$", "", q)
    q = re.sub(r"^(请问|请问一下)[，,\s]*", "", q)
    return q.strip(" ，,。？?：:") or query


def _compose_library_answer(results: list[dict[str, Any]], question: str) -> str:
    if not results:
        return (
            "现有文献库中没有检索到足够相关的证据。\n\n"
            "你可以先在上传页导入相关 PDF，或换用更具体的关键词再试一次。"
        )
    from app.services.rag import (  # 延迟导入，避免模块初始化顺序问题
        _document_frequencies,
        _lexical_overlap,
        _query_terms,
    )

    terms = _query_terms(question)
    documents = [str(item.get("content") or "") for item in results]
    idf = _document_frequencies(terms, documents)
    lines = [
        "根据文献库中检索到的原文片段，可以确认以下信息：",
        "",
    ]
    for i, item in enumerate(results[:6], start=1):
        content = collapse_ws(str(item.get("content") or ""))
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s+", content) if s.strip()]
        if len(sentences) > 3:
            scored = sorted(
                enumerate(sentences),
                key=lambda pair: _lexical_overlap(terms, pair[1], idf),
                reverse=True,
            )[:3]
            chosen = sorted(scored, key=lambda pair: pair[0])
            snippet = " ".join(s for _, s in chosen)
        else:
            snippet = content
        snippet = truncate(snippet, 650, "…")
        title = item.get("paper_title") or "未知论文"
        page = item.get("page_start", "?")
        lines.append(f"{i}. **{title}**（第 {page} 页）：{snippet} [{i}]")
    lines.append("")
    lines.append(
        "以上内容来自本地知识库的原文检索。若需要更精确的结论，可以指定论文标题或章节继续追问。"
    )
    return "\n".join(lines)


def _compose_search_answer(results: list[dict[str, Any]], question: str) -> str:
    if not results:
        return "没有找到匹配的论文。可以尝试更换关键词，或开启在线论文检索（AGENT_ONLINE_SEARCH=true）。"
    lines = [f"围绕「{question}」，文献库中找到 {len(results)} 篇相关论文：", ""]
    for i, item in enumerate(results[:5], start=1):
        authors = item.get("authors") or []
        author_text = "、".join(authors[:3]) + (" 等" if len(authors) > 3 else "")
        abstract = truncate(collapse_ws(str(item.get("abstract") or "")), 180)
        lines.append(
            f"{i}. **{item.get('title', '未知标题')}**"
            f"（{author_text or '作者未知'}，{item.get('year') or '年份未知'}）"
        )
        if abstract:
            lines.append(f"   {abstract}")
    lines.append("")
    lines.append("你可以指定其中一篇，让我继续分析它的方法、实验或结论。")
    return "\n".join(lines)


class MockLLM:
    """可预测的离线策略：仍然走真实的工具调用循环，便于无 Key 演示与 CI。"""

    provider = "mock"
    model = "wenlan-demo"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        started = time.perf_counter()
        question = _last_user(messages)
        results = _tool_results(messages)
        available = {t.get("function", {}).get("name") for t in (tools or [])}

        if results:
            last = results[-1]
            payload = last.get("payload") or {}
            if "results" in payload:
                items = payload.get("results") or []
                if last["name"] == "search_library":
                    content = _compose_library_answer(items, question)
                elif last["name"] == "get_paper_detail":
                    content = _compose_search_answer(
                        [payload.get("paper") or payload], question
                    )
                else:
                    content = _compose_search_answer(items, question)
            elif "paper" in payload:
                content = _compose_search_answer([payload["paper"]], question)
            elif payload.get("error"):
                content = f"工具调用没有成功：{payload['error']}。请换一种问法或稍后再试。"
            else:
                content = _compose_library_answer([], question)
            return LLMResult(
                content=content,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
                provider=self.provider,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        lower = question.lower().strip()
        if not lower or any(h in lower for h in _GREETING_HINTS) and len(lower) < 30:
            content = (
                "你好，我是文澜。可以帮你：\n\n"
                "1. 检索并发现论文；\n"
                "2. 查看论文作者、年份、摘要等详情；\n"
                "3. 基于本地 PDF 知识库进行多轮问答，并给出原文引用；\n"
                "4. 运行 Recall@K、工具调用准确率与任务成功率评测。\n\n"
                "试试问我：“RAG 论文中的生成器用了什么模型？”"
            )
            return LLMResult(
                content=content,
                model=self.model,
                provider=self.provider,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # 明确的本地 UUID
        uuid_match = re.search(r"\b([0-9a-f]{32})\b", lower)
        if uuid_match and "get_paper_detail" in available:
            call = ToolCall(
                id="call_mock_detail",
                name="get_paper_detail",
                arguments={"paper_id": uuid_match.group(1)},
            )
            return self._tool_result(call, started)

        metadata_intent = any(
            h in lower for h in ("作者", "年份", "详情", "发表在", "期刊", "doi", "引用格式")
        )
        content_intent = any(h in lower for h in _CONTENT_HINTS)
        search_intent = any(h in lower for h in _SEARCH_HINTS)
        explicit_search = any(h in lower for h in ("帮我找", "找关于", "搜索", "推荐", "检索论文", "检索一下"))
        behavior_intent = "没有证据" in lower or "没有相关" in lower

        if behavior_intent:
            return LLMResult(
                content=(
                    "如果文献库中没有足够证据，我会直接说明“现有文献库中没有足够证据”，"
                    "并建议补充 PDF 或换用关键词；不会编造作者、数字或结论。"
                ),
                model=self.model,
                provider=self.provider,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        if metadata_intent and "search_papers" in available:
            call = ToolCall(
                id="call_mock_search_meta",
                name="search_papers",
                arguments={"query": _clean_query(question), "limit": 3},
            )
            return self._tool_result(call, started)

        if explicit_search and "search_papers" in available:
            call = ToolCall(
                id="call_mock_search",
                name="search_papers",
                arguments={"query": _clean_query(question), "limit": 5},
            )
            return self._tool_result(call, started)

        if content_intent or not search_intent:
            if "search_library" in available:
                call = ToolCall(
                    id="call_mock_rag",
                    name="search_library",
                    arguments={"query": _clean_query(question), "top_k": 8},
                )
                return self._tool_result(call, started)

        if "search_papers" in available:
            call = ToolCall(
                id="call_mock_search",
                name="search_papers",
                arguments={"query": _clean_query(question), "limit": 5},
            )
            return self._tool_result(call, started)

        return LLMResult(
            content="当前没有可用的检索工具。请检查服务配置。",
            model=self.model,
            provider=self.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _tool_result(self, call: ToolCall, started: float) -> LLMResult:
        return LLMResult(
            content="",
            tool_calls=[call],
            model=self.model,
            provider=self.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        result = self.complete(messages)
        text = result.content
        # 按自然段切片，模拟流式输出，避免逐字刷屏。
        for i in range(0, len(text), 28):
            yield text[i : i + 28]
            time.sleep(0.008)


class LLMService:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or resolve_llm_config(settings)
        if self.config.is_mock:
            self.impl: OpenAICompatibleLLM | MockLLM = MockLLM()
        else:
            self.impl = OpenAICompatibleLLM(self.config)

    @property
    def is_mock(self) -> bool:
        return self.config.is_mock

    @property
    def label(self) -> str:
        return self.config.label

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        return self.impl.complete(messages, tools, temperature)

    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        return self.impl.stream(messages, temperature)

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "label": self.config.label,
            "is_mock": self.config.is_mock,
            "base_url": self.config.base_url if not self.config.is_mock else "",
            "configured": not self.config.is_mock,
        }


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service() -> None:
    global _llm_service
    _llm_service = None
