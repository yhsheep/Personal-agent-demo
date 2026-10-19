"""多轮对话与 SSE 流式问答 API。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db, session_scope
from app.models import Conversation, Message
from app.schemas import ChatRequest, ConversationCreate, ConversationUpdate
from app.services.agent import (
    AgentError,
    conversation_to_dict,
    get_agent_service,
    message_to_dict,
)
from app.services.llm import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
agent = get_agent_service()


@router.get("/conversations")
def list_conversations(limit: int = 50, db: Session = Depends(get_db)) -> dict:
    convs = db.scalars(
        select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
    ).all()
    items = []
    for conv in convs:
        last = db.scalars(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        ).first()
        items.append(conversation_to_dict(conv, last.content[:120] if last else ""))
    return {"items": items}


@router.post("/conversations")
def create_conversation(body: ConversationCreate, db: Session = Depends(get_db)) -> dict:
    conv = agent.create_conversation(db, title=body.title or "新的科研对话")
    db.commit()
    return {"conversation": conversation_to_dict(conv)}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> dict:
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    ).all()
    return {
        "conversation": conversation_to_dict(conv),
        "messages": [message_to_dict(m) for m in messages],
    }


@router.patch("/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str, body: ConversationUpdate, db: Session = Depends(get_db)
) -> dict:
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    conv.title = body.title.strip()
    db.commit()
    return {"conversation": conversation_to_dict(conv)}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> dict:
    conv = db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(conv)
    db.commit()
    return {"deleted": True, "conversation_id": conversation_id}


@router.post("/chat")
def chat(body: ChatRequest, db: Session = Depends(get_db)) -> dict:
    try:
        result = agent.chat(
            db,
            body.message,
            conversation_id=body.conversation_id,
            top_k=body.top_k,
            prompt_variant=body.prompt_variant,
        )
        db.commit()
        return result
    except AgentError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    except Exception as exc:
        db.rollback()
        logger.exception("对话失败")
        raise HTTPException(status_code=500, detail=f"对话失败：{exc}") from exc


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    if body.conversation_id:
        conv = db.get(Conversation, body.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")
        conversation_id = conv.id
    else:
        conv = agent.create_conversation(db)
        db.commit()
        conversation_id = conv.id

    def generator() -> Iterator[str]:
        try:
            with session_scope() as session:
                for item in agent.stream_chat(
                    session,
                    body.message,
                    conversation_id=conversation_id,
                    top_k=body.top_k,
                    prompt_variant=body.prompt_variant,
                ):
                    yield _sse(item["event"], item["data"])
        except Exception as exc:  # 生成器内异常必须转成 SSE，避免连接无响应
            logger.exception("SSE 流异常")
            yield _sse("error", {"message": f"流式响应异常：{exc}"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
