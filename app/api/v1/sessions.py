"""会话管理 API

提供会话的 CRUD 操作和标题生成功能。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request as StarletteRequest

from app.auth.middleware import TenantIdDep
from app.infra.database import get_session
from app.models.database import ChatSession
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.session import (
    GenerateTitleRequest,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
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
    "/{session_id}/generate-title",
    response_model=SessionResponse,
    summary="生成会话标题",
    description="基于会话的前几条消息自动生成标题",
)
@limiter.limit(RateLimit.API)
async def generate_title(
    request: StarletteRequest,
    session_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    data: GenerateTitleRequest | None = None,
    tenant_id: TenantIdDep = None,
) -> SessionResponse:
    """生成会话标题"""
    user_id = getattr(request.state, "user_id", None)

    service = SessionService(session)
    model_name = data.model_name if data else None

    title = await service.generate_title(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
        model_name=model_name,
    )

    # 获取更新后的会话
    chat_session = await service.get_session_or_404(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    message_count = await service.get_message_count(session_id)
    logger.info("session_title_generated_api", session_id=session_id, title=title)
    return _convert_to_response(chat_session, message_count)
