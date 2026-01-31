"""聊天 API 路由

提供聊天接口，支持同步和流式响应（SSE）。
遵循 LangGraph streaming 最佳实践。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request as StarletteRequest

from app.agent import get_agent
from app.agent.state import create_state_from_input
from app.config.settings import get_settings
from app.rate_limit.limiter import RateLimit, limiter
from app.agent.memory.context import get_context_manager
from app.auth.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.services.database import session_repository, session_scope

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ============== Request/Response Models ==============


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str = Field(..., description="用户消息", min_length=1)
    session_id: str = Field(..., description="会话 ID")
    user_id: str | None = Field(None, description="用户 ID")


class StreamChatRequest(ChatRequest):
    """流式聊天请求"""

    stream_mode: str = Field(
        "messages", description="流式模式: messages(令牌级), updates(状态更新), values(完整状态)"
    )


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


# ============== SSE Event Models ==============


class SSEEvent(BaseModel):
    """SSE 事件模型"""

    event: str = Field(default="message", description="事件类型")
    data: dict[str, Any] = Field(..., description="事件数据")

    def format(self) -> str:
        """格式化为 SSE 格式

        Returns:
            SSE 格式字符串
        """
        data_str = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\ndata: {data_str}\n\n"

async def _validate_session_access(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
) -> None:
    """验证会话是否存在，并可选校验用户/租户归属"""
    async with session_scope() as session:
        repo = session_repository(session)
        session_obj = await repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None:
            try:
                user_id_int = int(user_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid user_id") from exc

            if session_obj.user_id != user_id_int:
                raise HTTPException(status_code=403, detail="Session access denied")

        if tenant_id is not None and session_obj.tenant_id is not None:
            if session_obj.tenant_id != tenant_id:
                raise HTTPException(status_code=403, detail="Session tenant mismatch")

# ============== Chat Endpoints ==============


@router.post("", response_model=ChatResponse)
@limiter.limit(RateLimit.CHAT)
async def chat(
    request: ChatRequest,
    http_request: StarletteRequest,
    tenant_id: TenantIdDep = None,
) -> ChatResponse:
    """聊天接口（非流式）

    Args:
        request: 聊天请求

    Returns:
        ChatResponse: 聊天响应
    """
    try:
        effective_user_id = request.user_id
        state_user_id = getattr(http_request.state, "user_id", None)
        if (
            state_user_id is not None
            and effective_user_id is not None
            and str(state_user_id) != str(effective_user_id)
        ):
            raise HTTPException(status_code=403, detail="User mismatch")
        if state_user_id is not None and effective_user_id is None:
            effective_user_id = str(state_user_id)

        await _validate_session_access(
            session_id=request.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
        )
        agent = await get_agent()

        messages = await agent.get_response(
            message=request.message,
            session_id=request.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stream")
@limiter.limit(RateLimit.CHAT_STREAM)
async def chat_stream(
    request: StreamChatRequest,
    http_request: StarletteRequest,
    tenant_id: TenantIdDep = None,
) -> StreamingResponse:
    """流式聊天接口（SSE）

    遵循 LangGraph streaming 最佳实践：
    - 使用 Agent 层统一处理（包含 checkpointer、中间件、追踪）
    - 支持 stream_mode="messages" 获取令牌级流式输出
    - 返回 (message_chunk, metadata) 元组
    - 支持多种流式模式

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse: SSE 流式响应

    Examples:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/chat/stream",
            json={"message": "你好", "session_id": "test-123"},
            stream=True
        )

        for line in response.iter_lines():
            if line:
                print(line.decode())
        ```
    """

    effective_user_id = request.user_id
    state_user_id = getattr(http_request.state, "user_id", None)
    if (
        state_user_id is not None
        and effective_user_id is not None
        and str(state_user_id) != str(effective_user_id)
    ):
        raise HTTPException(status_code=403, detail="User mismatch")
    if state_user_id is not None and effective_user_id is None:
        effective_user_id = str(state_user_id)

    await _validate_session_access(
        session_id=request.session_id,
        user_id=effective_user_id,
        tenant_id=tenant_id,
    )

    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件流"""
        try:
            agent = await get_agent()
            graph = await agent.get_compiled_graph()

            # 准备输入
            input_data = create_state_from_input(
                input_text=request.message,
                user_id=effective_user_id,
                session_id=request.session_id,
            )

            # 准备配置
            from langgraph.types import RunnableConfig

            callbacks = []
            try:
                from app.agent.callbacks import KikiCallbackHandler

                settings = get_settings()
                callbacks.append(
                    KikiCallbackHandler(
                        session_id=request.session_id,
                        user_id=effective_user_id,
                        enable_langfuse=settings.langfuse_enabled,
                        enable_metrics=True,
                    )
                )
            except Exception:
                pass

            config = RunnableConfig(
                configurable={"thread_id": request.session_id},
                metadata={
                    "user_id": effective_user_id,
                    "session_id": request.session_id,
                    "tenant_id": tenant_id,
                },
                callbacks=callbacks or None,
            )

            logger.info(
                "sse_stream_start",
                session_id=request.session_id,
                stream_mode=request.stream_mode,
            )

            # 根据流式模式选择不同的处理方式
            if request.stream_mode == "messages":
                # 令牌级流式输出 (最佳实践)
                token_buffer: list[str] = []
                async for chunk, metadata in graph.astream(
                    input_data,
                    config,
                    stream_mode="messages",
                ):
                    if hasattr(chunk, "content") and chunk.content:
                        token_buffer.append(chunk.content)
                        event = SSEEvent(
                            event="token",
                            data={
                                "content": chunk.content,
                                "session_id": request.session_id,
                                "metadata": {
                                    "langgraph_node": metadata.get("langgraph_node"),
                                    "run_id": metadata.get("run_id"),
                                },
                            },
                        )
                        yield event.format()

                # 写入上下文存储（不影响主流程）
                if token_buffer:
                    await agent.persist_interaction(
                        request.session_id,
                        request.message,
                        "".join(token_buffer),
                    )

            elif request.stream_mode == "updates":
                # 状态更新流式输出
                async for chunk in graph.astream(
                    input_data,
                    config,
                    stream_mode="updates",
                ):
                    event = SSEEvent(
                        event="update",
                        data={
                            "update": chunk,
                            "session_id": request.session_id,
                        },
                    )
                    yield event.format()

            elif request.stream_mode == "values":
                # 完整状态流式输出
                async for chunk in graph.astream(
                    input_data,
                    config,
                    stream_mode="values",
                ):
                    event = SSEEvent(
                        event="state",
                        data={
                            "state": chunk,
                            "session_id": request.session_id,
                        },
                    )
                    yield event.format()

            # 发送完成事件
            done_event = SSEEvent(
                event="done",
                data={
                    "session_id": request.session_id,
                    "done": True,
                },
            )
            yield done_event.format()

            logger.info("sse_stream_complete", session_id=request.session_id)

        except Exception as e:
            logger.exception("sse_stream_failed", session_id=request.session_id)
            error_event = SSEEvent(
                event="error",
                data={
                    "error": str(e),
                    "session_id": request.session_id,
                    "done": True,
                },
            )
            yield error_event.format()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ============== History & Context Endpoints ==============


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
@limiter.limit(RateLimit.API)
async def get_chat_history(
    request: StarletteRequest,
    session_id: str,
    tenant_id: TenantIdDep = None,
) -> ChatHistoryResponse:
    """获取聊天历史

    Args:
        session_id: 会话 ID

    Returns:
        ChatHistoryResponse: 聊天历史
    """
    try:
        request_user_id = getattr(request.state, "user_id", None)
        await _validate_session_access(
            session_id=session_id,
            user_id=str(request_user_id) if request_user_id is not None else None,
            tenant_id=tenant_id,
        )
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/history/{session_id}")
@limiter.limit(RateLimit.API)
async def clear_chat_history(
    session_id: str,
    request: StarletteRequest,
    tenant_id: TenantIdDep = None,
) -> dict[str, str]:
    """清除聊天历史

    Args:
        session_id: 会话 ID
        request: FastAPI 请求对象

    Returns:
        操作结果
    """
    try:
        request_user_id = getattr(request.state, "user_id", None)
        await _validate_session_access(
            session_id=session_id,
            user_id=str(request_user_id) if request_user_id is not None else None,
            tenant_id=tenant_id,
        )
        agent = await get_agent()
        await agent.clear_chat_history(session_id)

        # 同时清除 Redis 上下文
        context_manager = get_context_manager()
        await context_manager.clear_context(session_id)

        return {"status": "success", "message": "聊天历史已清除"}

    except Exception as e:
        logger.exception("clear_chat_history_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/context/{session_id}/stats", response_model=ContextStatsResponse)
@limiter.limit(RateLimit.API)
async def get_context_stats(
    session_id: str,
    request: StarletteRequest,
    tenant_id: TenantIdDep = None,
) -> ContextStatsResponse:
    """获取会话上下文统计

    Args:
        session_id: 会话 ID

    Returns:
        ContextStatsResponse: 上下文统计
    """
    try:
        request_user_id = getattr(request.state, "user_id", None)
        await _validate_session_access(
            session_id=session_id,
            user_id=str(request_user_id) if request_user_id is not None else None,
            tenant_id=tenant_id,
        )
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/context/{session_id}")
@limiter.limit(RateLimit.API)
async def clear_context(
    session_id: str,
    request: StarletteRequest,
    tenant_id: TenantIdDep = None,
) -> dict[str, str]:
    """清除会话上下文（Redis 缓存）

    Args:
        session_id: 会话 ID

    Returns:
        操作结果
    """
    try:
        request_user_id = getattr(request.state, "user_id", None)
        await _validate_session_access(
            session_id=session_id,
            user_id=str(request_user_id) if request_user_id is not None else None,
            tenant_id=tenant_id,
        )
        context_manager = get_context_manager()
        await context_manager.clear_context(session_id)

        return {"status": "success", "message": "会话上下文已清除"}

    except Exception as e:
        logger.exception("clear_context_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
