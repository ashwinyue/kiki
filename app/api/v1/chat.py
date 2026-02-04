"""聊天 API 路由

提供聊天接口，支持同步和流式响应（SSE）。
使用 DeerFlow 风格的 Agent 系统（CompiledStateGraph）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import RunnableConfig
from starlette.requests import Request as StarletteRequest

from app.agent.graph import create_agent
from app.agent.graph.chat_stream import get_chat_stream_manager
from app.agent.message_utils import extract_ai_content
from app.agent.state import create_state_from_input
from app.api.v1.dependencies import (
    TenantIdDep,
    UserIdDep,
)
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Message,
    SSEEvent,
    StreamChatRequest,
)
from app.services.core.session_service import resolve_effective_user_id

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _validate_session_access_dep(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
) -> None:
    """验证会话访问权限（依赖注入）

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID

    Raises:
        HTTPException: 会话不存在或无权访问
    """
    from app.infra.database import session_scope
    from app.services.core.session_service import SessionService

    async with session_scope() as db_session:
        service = SessionService(db_session)
        await service.validate_session_access(
            session_id,
            user_id=int(user_id) if user_id else None,
            tenant_id=tenant_id,
        )


def _prepare_websearch_config(
    websearch,
) -> dict[str, str | int] | None:
    """准备 Web 搜索配置

    Args:
        websearch: Web 搜索配置

    Returns:
        配置字典或 None
    """
    if websearch and websearch.enable:
        return {
            "provider": websearch.provider,
            "search_depth": websearch.search_depth,
            "max_results": websearch.max_results,
        }
    return None


@router.post("", response_model=ChatResponse)
@limiter.limit(RateLimit.CHAT)
async def chat(
    request: ChatRequest,
    http_request: StarletteRequest,
    tenant_id: TenantIdDep = None,
    user_id: UserIdDep = None,
) -> ChatResponse:
    """聊天接口（非流式）

    使用 DeerFlow 风格 Agent 系统：
    - 动态创建 Agent（CompiledStateGraph）
    - 支持 stream_mode="messages" 流式输出
    - Checkpoint 持久化

    Args:
        request: 聊天请求
        http_request: FastAPI 请求对象（用于限流）
        tenant_id: 租户 ID（依赖注入）
        user_id: 用户 ID（依赖注入）

    Returns:
        ChatResponse: 聊天响应
    """
    try:
        effective_user_id = resolve_effective_user_id(request.user_id, user_id)

        await _validate_session_access_dep(
            session_id=request.session_id,
            user_id=effective_user_id,
            tenant_id=tenant_id,
        )

        # 创建 Agent（DeerFlow 风格）
        agent = create_agent(
            agent_name="chat",
            agent_type="chat",
            prompt_template="chat_with_tools",
        )

        # 准备输入
        input_data = create_state_from_input(
            input_text=request.message,
            user_id=effective_user_id,
            session_id=request.session_id,
        )

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": request.session_id},
        )

        # 执行 Agent
        result = await agent.ainvoke(input_data, config)

        # 提取响应内容
        content = extract_ai_content(result.get("messages", []))

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
    user_id: UserIdDep = None,
) -> StreamingResponse:
    """流式聊天接口（SSE）

    使用 DeerFlow 风格 Agent 系统：
    - 动态创建 Agent（CompiledStateGraph）
    - 使用 LangGraph streaming API
    - 支持 stream_mode="messages" 令牌级输出
    - Checkpoint 持久化 + ChatStreamManager

    Args:
        request: 聊天请求
        http_request: FastAPI 请求对象（用于限流）
        tenant_id: 租户 ID（依赖注入）
        user_id: 用户 ID（依赖注入）

    Returns:
        StreamingResponse: SSE 流式响应
    """
    # 解析有效用户 ID
    effective_user_id = resolve_effective_user_id(request.user_id, user_id)

    # 验证会话访问权限
    await _validate_session_access_dep(
        session_id=request.session_id,
        user_id=effective_user_id,
        tenant_id=tenant_id,
    )

    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件流"""
        try:
            # 创建 Agent（DeerFlow 风格）
            agent = create_agent(
                agent_name="chat",
                agent_type="chat",
                prompt_template="chat_with_tools",
            )

            # 准备输入
            input_data = create_state_from_input(
                input_text=request.message,
                user_id=effective_user_id,
                session_id=request.session_id,
            )

            # 准备配置
            callbacks = []
            try:
                from app.agent.callbacks import KikiCallbackHandler
                from app.config.settings import get_settings

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
                async for chunk, metadata in agent.astream(
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

                # 持久化到 ChatStreamManager
                stream_manager = await get_chat_stream_manager()
                await stream_manager.process_stream_message(
                    thread_id=request.session_id,
                    message="".join(token_buffer),
                    finish_reason="stop",
                )

            elif request.stream_mode == "updates":
                # 状态更新流式输出
                async for chunk in agent.astream(
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
                async for chunk in agent.astream(
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
    session_id: str,
    http_request: StarletteRequest,
    tenant_id: TenantIdDep = None,
    user_id: UserIdDep = None,
) -> ChatHistoryResponse:
    """获取聊天历史

    使用 ChatStreamManager 获取持久化的聊天历史。

    Args:
        session_id: 会话 ID
        tenant_id: 租户 ID（依赖注入）
        user_id: 用户 ID（依赖注入）

    Returns:
        ChatHistoryResponse: 聊天历史
    """
    try:
        await _validate_session_access_dep(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        stream_manager = await get_chat_stream_manager()
        messages = await stream_manager.get_chat_history(session_id)

        if not messages:
            return ChatHistoryResponse(messages=[], session_id=session_id)

        # 转换为响应格式
        history_messages = []
        for msg in messages:
            history_messages.append(
                Message(
                    role="assistant",
                    content=msg,
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
    tenant_id: TenantIdDep = None,
    user_id: UserIdDep = None,
) -> dict[str, str]:
    """清除聊天历史

    使用 ChatStreamManager 删除持久化的聊天历史。

    Args:
        session_id: 会话 ID
        tenant_id: 租户 ID（依赖注入）
        user_id: 用户 ID（依赖注入）

    Returns:
        操作结果
    """
    try:
        await _validate_session_access_dep(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # 删除聊天历史
        stream_manager = await get_chat_stream_manager()
        await stream_manager.delete_chat_history(session_id)

        return {"status": "success", "message": "聊天历史已清除"}

    except Exception as e:
        logger.exception("clear_chat_history_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
