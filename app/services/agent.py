"""Tool Calling Agent：规划、执行工具、生成带证据的回答、持久化多轮会话。"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Conversation, Message
from app.services.llm import LLMError, LLMResult, LLMService, get_llm_service
from app.services.paper_sources import PaperSearchService
from app.services.prompts import PromptVariant, get_prompt_variant
from app.services.rag import RAGService
from app.services.tools import ToolContext, ToolRegistry, compact_tool_result
from app.utils.text import estimate_tokens, truncate

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    pass


@dataclass
class PlanOutcome:
    messages: list[dict[str, Any]]
    content: str
    tool_trace: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    usage: dict[str, int]
    llm_latency_ms: int
    raw_results: list[dict[str, Any]] = field(default_factory=list)


def new_title(question: str) -> str:
    title = re.sub(r"\s+", " ", question or "").strip()
    title = re.sub(r"^(请|麻烦|帮我|我想|能否|可以)?\s*(帮我|帮忙)?\s*", "", title)
    return truncate(title, 36, "…") or "新的科研对话"


def _sum_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in items:
        for key in total:
            total[key] += int(item.get(key) or 0)
    return total


def _add_usage(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(base.get(key) or 0) + int(extra.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _chunks(text: str, size: int = 32) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


class AgentService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or get_llm_service()
        self.search = PaperSearchService()
        self.rag = RAGService()

    # ------------------------------------------------------------------
    # 会话准备
    # ------------------------------------------------------------------
    def get_conversation(self, session: Session, conversation_id: str) -> Conversation:
        conv = session.get(Conversation, conversation_id)
        if not conv:
            raise AgentError("会话不存在")
        return conv

    def create_conversation(self, session: Session, title: str = "新的科研对话") -> Conversation:
        conv = Conversation(title=title)
        session.add(conv)
        session.flush()
        return conv

    def _history(
        self, session: Session, conversation_id: str, limit_turns: int | None = None
    ) -> list[dict[str, str]]:
        limit = (limit_turns or settings.agent_history_turns) * 2
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            ).all()
        )
        messages.reverse()
        history: list[dict[str, str]] = []
        for msg in messages:
            if msg.role in {"user", "assistant"} and msg.content:
                history.append({"role": msg.role, "content": msg.content})
        return history

    def _system_prompt(self, variant: PromptVariant, citation_count: int = 0) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        extra = (
            "\n工具返回的 citation_index 是跨多次检索的全局编号。"
            "回答中的 [n] 必须与这些编号严格对应。"
        )
        if citation_count:
            extra += f"\n本轮已累计 {citation_count} 条证据，请优先使用它们。"
        return f"{variant.system}\n当前日期：{today}。{extra}"

    # ------------------------------------------------------------------
    # 工具规划循环
    # ------------------------------------------------------------------
    def _plan(
        self,
        session: Session,
        conversation_id: str,
        history: list[dict[str, str]],
        question: str,
        variant: PromptVariant,
        top_k: int | None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PlanOutcome:
        ctx = ToolContext(
            session=session,
            rag=self.rag,
            search=self.search,
            on_event=on_event,
            default_top_k=top_k or settings.top_k,
        )
        registry = ToolRegistry(ctx)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(variant)}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        trace: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []
        usage_items: list[dict[str, Any]] = []
        llm_latency = 0
        seen_calls: set[str] = set()

        for round_index in range(1, settings.agent_max_tool_rounds + 1):
            result: LLMResult = self.llm.complete(
                messages,
                tools=registry.schemas(),
                temperature=variant.temperature,
            )
            usage_items.append(result.usage)
            llm_latency += result.latency_ms
            if not result.tool_calls:
                return PlanOutcome(
                    messages=messages,
                    content=result.content,
                    tool_trace=trace,
                    citations=ctx.citations,
                    usage=_sum_usage(usage_items),
                    llm_latency_ms=llm_latency,
                    raw_results=raw_results,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [tc.as_openai() for tc in result.tool_calls],
                }
            )
            for call in result.tool_calls:
                signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                if signature in seen_calls:
                    tool_result: dict[str, Any] = {
                        "error": "重复调用相同工具与参数，请基于已有结果回答。",
                        "ok": False,
                    }
                    status = "skipped"
                else:
                    seen_calls.add(signature)
                    tool_result = registry.execute(call.name, call.arguments)
                    status = "success" if tool_result.get("ok") else "error"
                tool_result.setdefault("latency_ms", 0)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                raw_results.append({"name": call.name, "result": tool_result})
                trace.append(
                    {
                        "round": round_index,
                        "name": call.name,
                        "arguments": call.arguments,
                        "status": status,
                        "latency_ms": tool_result.get("latency_ms", 0),
                        "summary": ToolRegistry._summary(call.name, tool_result),
                        "result": compact_tool_result(call.name, tool_result),
                    }
                )

        # 轮次耗尽，要求模型基于已有证据收束
        messages.append(
            {
                "role": "system",
                "content": "工具调用轮次已达上限。请立即基于以上已获得的结果回答；证据不足时明确说明。",
            }
        )
        result = self.llm.complete(messages, tools=None, temperature=variant.temperature)
        usage_items.append(result.usage)
        llm_latency += result.latency_ms
        content = result.content or (
            "工具调用轮次已达上限，且未能形成可靠结论。请缩小问题范围后重试。"
        )
        return PlanOutcome(
            messages=messages,
            content=content,
            tool_trace=trace,
            citations=ctx.citations,
            usage=_sum_usage(usage_items),
            llm_latency_ms=llm_latency,
            raw_results=raw_results,
        )

    # ------------------------------------------------------------------
    # 引用处理与持久化
    # ------------------------------------------------------------------
    @staticmethod
    def _select_citations(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs = {int(x) for x in re.findall(r"\[(\d{1,2})\]", answer or "")}
        out: list[dict[str, Any]] = []
        for c in citations:
            item = dict(c)
            item["cited"] = c.get("index") in refs
            out.append(item)
        if refs:
            cited = [c for c in out if c["cited"]]
            if cited:
                return cited + [c for c in out if not c["cited"]]
        return out

    def _save_message(
        self,
        session: Session,
        conversation: Conversation,
        role: str,
        content: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        latency_ms: int = 0,
    ) -> Message:
        msg = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            citations=citations or [],
            tool_trace=tool_trace or [],
            usage=usage or {},
            latency_ms=latency_ms,
        )
        session.add(msg)
        conversation.message_count = (conversation.message_count or 0) + 1
        conversation.updated_at = datetime.now(timezone.utc)
        if conversation.title in ("", "新的科研对话") and role == "user":
            conversation.title = new_title(content)
        session.flush()
        return msg

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------
    def chat(
        self,
        session: Session,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
        prompt_variant: str = "citation_strict",
    ) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise AgentError("问题不能为空")
        started = time.perf_counter()
        if conversation_id:
            conversation = self.get_conversation(session, conversation_id)
        else:
            conversation = self.create_conversation(session)
        history = self._history(session, conversation.id)
        self._save_message(session, conversation, "user", question)
        variant = get_prompt_variant(prompt_variant)

        plan = self._plan(
            session,
            conversation.id,
            history,
            question,
            variant,
            top_k,
        )
        if plan.content:
            answer = plan.content
        else:
            final_messages = list(plan.messages)
            final_messages.append(
                {
                    "role": "system",
                    "content": "请基于工具返回的证据用中文回答，保留 citation_index 对应的 [n] 引用。",
                }
            )
            try:
                result = self.llm.complete(
                    final_messages, tools=None, temperature=variant.temperature
                )
                answer = result.content
                plan.usage = _add_usage(plan.usage, result.usage)
                plan.llm_latency_ms += result.latency_ms
            except LLMError:
                if plan.raw_results:
                    answer = self._fallback_answer(plan.raw_results)
                else:
                    raise
        if not answer:
            answer = "模型没有返回有效内容，请重试。"
        citations = self._select_citations(answer, plan.citations)
        latency = int((time.perf_counter() - started) * 1000)
        assistant = self._save_message(
            session,
            conversation,
            "assistant",
            answer,
            citations=citations,
            tool_trace=plan.tool_trace,
            usage=plan.usage,
            latency_ms=latency,
        )
        return {
            "conversation_id": conversation.id,
            "conversation_title": conversation.title,
            "message": message_to_dict(assistant),
            "citations": citations,
            "tool_trace": plan.tool_trace,
            "usage": plan.usage,
            "latency_ms": latency,
            "llm": self.llm.status(),
            "prompt_variant": variant.key,
        }

    def answer_once(
        self,
        session: Session,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        prompt_variant: str = "citation_strict",
    ) -> dict[str, Any]:
        """评测/批处理使用：不创建会话，不写入 messages。"""
        variant = get_prompt_variant(prompt_variant)
        started = time.perf_counter()
        plan = self._plan(
            session,
            "",
            history or [],
            question,
            variant,
            top_k,
        )
        answer = plan.content
        if not answer:
            final_messages = list(plan.messages)
            final_messages.append(
                {
                    "role": "system",
                    "content": "请基于工具返回的证据用中文回答，保留 citation_index 对应的 [n] 引用。",
                }
            )
            result = self.llm.complete(
                final_messages, tools=None, temperature=variant.temperature
            )
            answer = result.content
            plan.usage = _add_usage(plan.usage, result.usage)
        citations = self._select_citations(answer, plan.citations)
        return {
            "answer": answer,
            "citations": citations,
            "tool_trace": plan.tool_trace,
            "usage": plan.usage,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    @staticmethod
    def _fallback_answer(raw_results: list[dict[str, Any]]) -> str:
        for item in raw_results:
            name = item.get("name")
            result = item.get("result") or {}
            if name == "search_library" and result.get("results"):
                parts = ["根据本地文献库检索到的原文：", ""]
                for r in result["results"][:3]:
                    parts.append(
                        f"- 《{r.get('paper_title')}》第 {r.get('page_start')} 页："
                        f"{truncate(r.get('content', ''), 280)} [{r.get('citation_index')}]"
                    )
                return "\n".join(parts)
            if name in {"search_papers", "search_online_papers"} and result.get("results"):
                return "找到以下论文：\n\n" + "\n".join(
                    f"- {r.get('title')}（{r.get('year') or '年份未知'}）"
                    for r in result["results"][:5]
                )
        return "检索已完成，但模型未能生成回答，请稍后重试。"

    # ------------------------------------------------------------------
    # 流式
    # ------------------------------------------------------------------
    def stream_chat(
        self,
        session: Session,
        question: str,
        *,
        conversation_id: str,
        top_k: int | None = None,
        prompt_variant: str = "citation_strict",
    ) -> Iterator[dict[str, Any]]:
        question = (question or "").strip()
        if not question:
            yield {"event": "error", "data": {"message": "问题不能为空"}}
            return
        started = time.perf_counter()
        try:
            conversation = self.get_conversation(session, conversation_id)
            history = self._history(session, conversation.id)
            self._save_message(session, conversation, "user", question)
            variant = get_prompt_variant(prompt_variant)

            queue: list[tuple[str, dict[str, Any]]] = []

            def on_event(event: str, data: dict[str, Any]) -> None:
                queue.append((event, data))

            yield {
                "event": "start",
                "data": {
                    "conversation_id": conversation.id,
                    "llm": self.llm.status(),
                    "prompt_variant": variant.key,
                },
            }
            plan = self._plan(
                session,
                conversation.id,
                history,
                question,
                variant,
                top_k,
                on_event,
            )
            # on_event 是同步回调，这里按顺序补发工具事件
            for event, data in queue:
                yield {"event": event, "data": data}

            answer = plan.content
            if not answer:
                final_messages = list(plan.messages)
                final_messages.append(
                    {
                        "role": "system",
                        "content": "请基于工具返回的证据用中文回答，保留 citation_index 对应的 [n] 引用。",
                    }
                )
                pieces: list[str] = []
                try:
                    for token in self.llm.stream(
                        final_messages, temperature=variant.temperature
                    ):
                        pieces.append(token)
                        yield {"event": "token", "data": {"text": token}}
                    answer = "".join(pieces)
                except LLMError as exc:
                    logger.warning("最终回答流式生成失败，使用降级回答：%s", exc)
                    answer = self._fallback_answer(plan.raw_results)
                    for piece in _chunks(answer):
                        yield {"event": "token", "data": {"text": piece}}
            else:
                for piece in _chunks(answer):
                    yield {"event": "token", "data": {"text": piece}}

            citations = self._select_citations(answer, plan.citations)
            if plan.usage.get("total_tokens", 0) == 0 and answer:
                plan.usage["completion_tokens"] = estimate_tokens(answer)
                plan.usage["total_tokens"] = plan.usage["completion_tokens"]
            latency = int((time.perf_counter() - started) * 1000)
            assistant = self._save_message(
                session,
                conversation,
                "assistant",
                answer,
                citations=citations,
                tool_trace=plan.tool_trace,
                usage=plan.usage,
                latency_ms=latency,
            )
            yield {"event": "citations", "data": {"citations": citations}}
            yield {
                "event": "done",
                "data": {
                    "conversation_id": conversation.id,
                    "conversation_title": conversation.title,
                    "message_id": assistant.id,
                    "citations": citations,
                    "tool_trace": plan.tool_trace,
                    "usage": plan.usage,
                    "latency_ms": latency,
                },
            }
        except AgentError as exc:
            yield {"event": "error", "data": {"message": str(exc)}}
        except LLMError as exc:
            logger.exception("Agent LLM 调用失败")
            yield {
                "event": "error",
                "data": {
                    "message": f"模型调用失败：{exc}",
                    "hint": "请检查 .env 中的 API Key、模型名与网络；也可设置 LLM_PROVIDER=mock 使用离线演示。",
                },
            }
        except Exception as exc:
            logger.exception("Agent 执行失败")
            yield {"event": "error", "data": {"message": f"Agent 执行失败：{exc}"}}


def message_to_dict(msg: Message) -> dict[str, Any]:
    return {
        "id": msg.id,
        "conversation_id": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "citations": msg.citations or [],
        "tool_trace": msg.tool_trace or [],
        "usage": msg.usage or {},
        "latency_ms": msg.latency_ms,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


def conversation_to_dict(conv: Conversation, last_message: str = "") -> dict[str, Any]:
    return {
        "id": conv.id,
        "title": conv.title,
        "summary": conv.summary,
        "message_count": conv.message_count,
        "last_message": last_message,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


def reset_agent_service() -> None:
    global _agent_service
    _agent_service = None
