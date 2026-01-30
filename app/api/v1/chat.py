"""聊天 API 路由

提供聊天接口，支持同步和流式响应。
"""

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.agent import get_agent
from app.core.logging import get_logger
from app.core.memory import get_context_manager, ChatMessage


logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str = Field(..., description="用户消息", min_length=1)
    session_id: str = Field(..., description="会话 ID")
    user_id: str | None = Field(None, description="用户 ID")
    stream: bool = Field(False, description="是否使用流式响应")


class ContextStatsResponse(BaseModel):
    """上下文统计响应"""

    session_id: str
    message_count: int
    token_estimate: int
    role_distribution: dict[str, int]
    exists: bool


class ChatResponse(BaseModel):
    """聊天响应"""

    content: str = Field(..., description="响应内容")
    session_id: str = Field(..., description="会话 ID")


class Message(BaseModel):
    """消息"""

    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""

    messages: list[Message] = Field(default_factory=list, description="历史消息")
    session_id: str = Field(..., description="会话 ID")


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """聊天接口

    Args:
        request: 聊天请求

    Returns:
        ChatResponse: 聊天响应
    """
    try:
        agent = await get_agent()

        messages = await agent.get_response(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        # 获取最后一条 AI 消息
        content = ""
        for msg in reversed(messages):
            if msg.type == "ai":
                content = msg.content
                break

        return ChatResponse(content=content, session_id=request.session_id)

    except Exception as e:
        logger.exception("chat_request_failed", session_id=request.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse: 流式响应
    """

    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件"""
        try:
            agent = await get_agent()

            async for chunk in agent.get_stream_response(
                message=request.message,
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                yield f"data: {json.dumps({'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"

            # 发送完成信号
            yield f"data: {json.dumps({'content': '', 'done': True}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("chat_stream_failed", session_id=request.session_id)
            yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """获取聊天历史

    Args:
        session_id: 会话 ID

    Returns:
        ChatHistoryResponse: 聊天历史
    """
    try:
        agent = await get_agent()

        messages = await agent.get_chat_history(session_id)

        # 转换为响应格式
        history_messages = []
        for msg in messages:
            if msg.type in ("human", "ai", "system"):
                role_map = {"human": "user", "ai": "assistant", "system": "system"}
                history_messages.append(
                    Message(
                        role=role_map.get(msg.type, msg.type),
                        content=str(msg.content),
                    )
                )

        return ChatHistoryResponse(messages=history_messages, session_id=session_id)

    except Exception as e:
        logger.exception("get_chat_history_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> dict[str, str]:
    """清除聊天历史

    Args:
        session_id: 会话 ID

    Returns:
        操作结果
    """
    try:
        agent = await get_agent()
        await agent.clear_chat_history(session_id)

        # 同时清除 Redis 上下文
        context_manager = get_context_manager()
        await context_manager.clear_context(session_id)

        return {"status": "success", "message": "聊天历史已清除"}

    except Exception as e:
        logger.exception("clear_chat_history_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{session_id}/stats", response_model=ContextStatsResponse)
async def get_context_stats(session_id: str) -> ContextStatsResponse:
    """获取会话上下文统计

    Args:
        session_id: 会话 ID

    Returns:
        ContextStatsResponse: 上下文统计
    """
    try:
        context_manager = get_context_manager()
        stats = await context_manager.get_stats(session_id)

        return ContextStatsResponse(
            session_id=stats["session_id"],
            message_count=stats["message_count"],
            token_estimate=stats["token_estimate"],
            role_distribution=stats["role_distribution"],
            exists=stats["exists"],
        )

    except Exception as e:
        logger.exception("get_context_stats_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/context/{session_id}")
async def clear_context(session_id: str) -> dict[str, str]:
    """清除会话上下文（Redis 缓存）

    Args:
        session_id: 会话 ID

    Returns:
        操作结果
    """
    try:
        context_manager = get_context_manager()
        await context_manager.clear_context(session_id)

        return {"status": "success", "message": "会话上下文已清除"}

    except Exception as e:
        logger.exception("clear_context_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e))
