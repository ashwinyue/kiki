"""会话管理 API

提供会话的 CRUD 操作和标题生成功能。
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request as StarletteRequest

from app.infra.database import get_session
from app.middleware import TenantIdDep
from app.models.database import ChatSession
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.chat import SSEEvent
from app.schemas.session import (
    _DEFAULT_SESSION_STOP_REQUEST,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
    SessionStateResponse,
    SessionStopRequest,
    SessionStopResponse,
    SessionUpdate,
    StreamInfoResponse,
)
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = get_logger(__name__)


def _convert_to_response(session: ChatSession, message_count: int = 0) -> SessionResponse:
    """转换会话对象为响应模型

    Args:
        session: 会话对象
        message_count: 消息数量

    Returns:
        会话响应模型
    """
    return SessionResponse(
        id=session.id,
        name=session.name,
        user_id=session.user_id,
        tenant_id=session.tenant_id,
        agent_id=session.agent_id,
        message_count=message_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _convert_to_detail_response(session: ChatSession, message_count: int = 0) -> SessionDetailResponse:
    """转换会话对象为详情响应模型

    Args:
        session: 会话对象
        message_count: 消息数量

    Returns:
        会话详情响应模型
    """
    return SessionDetailResponse(
        id=session.id,
        name=session.name,
        user_id=session.user_id,
        tenant_id=session.tenant_id,
        agent_id=session.agent_id,
        message_count=message_count,
        agent_config=session.agent_config,
        context_config=session.context_config,
        extra_data=session.extra_data,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建会话",
    description="创建新的会话，自动创建对应的 Thread 用于 LangGraph 状态持久化",
)
@limiter.limit(RateLimit.API)
async def create_session(
    request: StarletteRequest,
    data: SessionCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> SessionResponse:
    """创建会话"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    chat_session = await service.create_session(
        data,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    logger.info("session_created_api", session_id=chat_session.id, name=chat_session.name)
    return _convert_to_response(chat_session)


@router.get(
    "",
    response_model=SessionListResponse,
    summary="获取会话列表",
    description="分页获取会话列表，支持按用户和租户筛选",
)
@limiter.limit(RateLimit.API)
async def list_sessions(
    request: StarletteRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> SessionListResponse:
    """获取会话列表"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    result = await service.list_sessions(
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
        page=page,
        size=size,
    )

    # 获取每个会话的消息数量
    items = []
    for chat_session in result.items:
        message_count = await service.get_message_count(chat_session.id)
        items.append(_convert_to_response(chat_session, message_count))

    return SessionListResponse(
        items=items,
        total=result.total,
        page=result.page,
        size=result.size,
        pages=result.pages,
    )


@router.get(
    "/{session_id}",
    response_model=SessionDetailResponse,
    summary="获取会话详情",
    description="根据 ID 获取会话的详细信息",
)
@limiter.limit(RateLimit.API)
async def get_session(
    request: StarletteRequest,
    session_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> SessionDetailResponse:
    """获取会话详情"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    chat_session = await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    message_count = await service.get_message_count(session_id)
    return _convert_to_detail_response(chat_session, message_count)


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="更新会话",
    description="更新会话的名称、配置等信息",
)
@limiter.limit(RateLimit.API)
async def update_session(
    request: StarletteRequest,
    session_id: str,
    data: SessionUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> SessionResponse:
    """更新会话"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    chat_session = await service.update_session(
        session_id,
        data,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    message_count = await service.get_message_count(session_id)
    logger.info("session_updated_api", session_id=session_id)
    return _convert_to_response(chat_session, message_count)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话",
    description="软删除指定的会话，同时归档对应的 Thread",
)
@limiter.limit(RateLimit.API)
async def delete_session(
    request: StarletteRequest,
    session_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> None:
    """删除会话"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    success = await service.delete_session(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在",
        )

    logger.info("session_deleted_api", session_id=session_id)


@router.post(
    "/{session_id}/stop",
    response_model=SessionStopResponse,
    summary="停止会话",
    description="中断正在进行的流式响应",
)
@limiter.limit(RateLimit.API)
async def stop_session(
    session_id: str,
    request: StarletteRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
    data: SessionStopRequest = _DEFAULT_SESSION_STOP_REQUEST,
) -> SessionStopResponse:
    """停止会话

    中断正在进行的流式响应。
    """
    from app.services.session_state import get_session_state_manager

    user_id = getattr(request.state, "user_id", None)

    # 验证会话存在
    service = SessionService(session)
    await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    # 获取状态管理器
    state_manager = get_session_state_manager()

    # 请求停止会话
    success = await state_manager.request_stop(session_id)

    current_state = await state_manager.get_state(session_id)

    if success:
        logger.info(
            "session_stopped_api",
            session_id=session_id,
            reason=data.reason,
            force=data.force,
        )
        return SessionStopResponse(
            session_id=session_id,
            stopped=True,
            state=current_state.state.value if current_state else "unknown",
            message="会话已停止",
        )
    else:
        return SessionStopResponse(
            session_id=session_id,
            stopped=False,
            state=current_state.state.value if current_state else "unknown",
            message="停止会话失败",
        )


@router.get(
    "/{session_id}/state",
    response_model=SessionStateResponse,
    summary="获取会话状态",
    description="获取会话的当前状态",
)
@limiter.limit(RateLimit.API)
async def get_session_state(
    session_id: str,
    request: StarletteRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> SessionStateResponse:
    """获取会话状态

    返回会话的当前运行状态。
    """
    from app.services.session_state import get_session_state_manager

    user_id = getattr(request.state, "user_id", None)

    # 验证会话存在
    service = SessionService(session)
    await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    # 获取状态
    state_manager = get_session_state_manager()
    state_info = await state_manager.get_state(session_id)

    from app.schemas.session import SessionStateEnum
    from app.services.session_state import SessionState

    if state_info is None:
        return SessionStateResponse(
            session_id=session_id,
            state=SessionStateEnum.IDLE,
            is_running=False,
            is_stopping=False,
        )

    # 将内部状态枚举转换为 API 枚举
    api_state = SessionStateEnum(state_info.state.value)

    return SessionStateResponse(
        session_id=session_id,
        state=api_state,
        is_running=state_info.state == SessionState.RUNNING,
        is_stopping=state_info.state in (SessionState.STOPPING, SessionState.STOPPED),
    )


@router.get(
    "/{session_id}/stream-info",
    response_model=StreamInfoResponse,
    summary="获取流信息",
    description="获取会话的流式处理状态信息",
)
@limiter.limit(RateLimit.API)
async def get_stream_info(
    session_id: str,
    request: StarletteRequest,
    db_session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
) -> StreamInfoResponse:
    """获取流信息

    返回会话的流式处理状态。
    """
    from app.agent.streaming.service import get_stream_continuation_service

    user_id = getattr(request.state, "user_id", None)

    # 验证会话存在
    service = SessionService(db_session)
    await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    # 获取流元数据
    stream_service = get_stream_continuation_service()
    metadata = await stream_service.get_metadata(session_id)

    if metadata is None:
        return StreamInfoResponse(
            session_id=session_id,
            is_active=False,
            event_count=0,
            started_at=None,
            updated_at=None,
        )

    return StreamInfoResponse(
        session_id=session_id,
        is_active=metadata.is_active,
        event_count=metadata.event_count,
        started_at=metadata.started_at,
        updated_at=metadata.updated_at,
    )


@router.get(
    "/{session_id}/continue-stream",
    summary="继续接收流",
    description="继续接收正在进行的流式响应，支持客户端重连",
)
@limiter.limit(RateLimit.CHAT_STREAM)
async def continue_stream(
    session_id: str,
    request: StarletteRequest,
    db_session: Annotated[AsyncSession, Depends(get_session)],
    tenant_id: TenantIdDep = None,
    since: int = Query(0, ge=0, description="从第几个事件开始"),
    wait_timeout: int = Query(60, ge=5, le=300, description="等待超时时间（秒）"),
) -> StreamingResponse:
    """继续接收流

    对齐 WeKnora 的 sessions/continue-stream/{id} API。

    支持客户端重连到正在进行的流式会话：
    - 如果会话正在流式处理，实时返回新事件
    - 如果会话已完成，返回缓存的事件
    - 支持从指定位置开始接收（since 参数）

    返回 SSE 流式响应。
    """
    from app.agent.streaming.service import get_stream_continuation_service

    user_id = getattr(request.state, "user_id", None)

    # 验证会话存在
    service = SessionService(db_session)
    await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    stream_service = get_stream_continuation_service()

    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件流"""
        try:
            # 检查流是否存在
            metadata = await stream_service.get_metadata(session_id)

            if metadata is None:
                # 流不存在，发送提示事件
                event = SSEEvent(
                    event="error",
                    data={
                        "error": "No active stream for this session",
                        "session_id": session_id,
                        "done": True,
                    },
                )
                yield event.format()
                return

            # 发送元数据事件
            meta_event = SSEEvent(
                event="metadata",
                data={
                    "session_id": session_id,
                    "is_active": metadata.is_active,
                    "event_count": metadata.event_count,
                    "since": since,
                },
            )
            yield meta_event.format()

            # 获取事件
            event_count = 0
            start_time = asyncio.get_event_loop().time()

            async for event in stream_service.get_events(session_id, since=since):
                # 转换为 SSE 事件
                sse_event = SSEEvent(
                    event=event.event_type,
                    data={
                        "content": event.content,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp,
                        "session_id": session_id,
                    },
                )
                yield sse_event.format()
                event_count += 1

                # 如果收到完成事件，结束流
                if event.event_type == "done":
                    break

                # 检查超时
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > wait_timeout:
                    logger.info(
                        "continue_stream_timeout",
                        session_id=session_id,
                        elapsed=elapsed,
                    )
                    timeout_event = SSEEvent(
                        event="timeout",
                        data={
                            "message": "Stream continuation timeout",
                            "session_id": session_id,
                            "done": True,
                        },
                    )
                    yield timeout_event.format()
                    break

                # 如果流仍然活跃，等待新事件
                if metadata.is_active:
                    await asyncio.sleep(0.1)
                    # 刷新元数据
                    metadata = await stream_service.get_metadata(session_id)
                    if metadata is None:
                        break
                else:
                    # 流已完成，结束
                    break

            logger.info(
                "continue_stream_complete",
                session_id=session_id,
                events_sent=event_count,
            )

        except Exception as e:
            logger.exception(
                "continue_stream_failed",
                session_id=session_id,
                error=str(e),
            )
            error_event = SSEEvent(
                event="error",
                data={
                    "error": str(e),
                    "session_id": session_id,
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
            "X-Accel-Buffering": "no",
        },
    )
