"""消息管理 API

提供消息的查询、编辑、删除和搜索功能。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request as StarletteRequest

from app.auth.middleware import TenantIdDep
from app.infra.database import get_session
from app.models.database import Message
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.schemas.message import (
    MessageListResponse,
    MessageRegenerateRequest,
    MessageResponse,
    MessageSearchResponse,
    MessageUpdate,
)
from app.services.message_service import MessageService

router = APIRouter(prefix="/messages", tags=["messages"])
logger = get_logger(__name__)


def _convert_to_response(message: Message) -> MessageResponse:
    """转换消息对象为响应模型

    Args:
        message: 消息对象

    Returns:
        消息响应模型
    """
    return MessageResponse(
        id=message.id,
        session_id=message.session_id or "",
        role=message.role,
        content=message.content,
        is_completed=message.is_completed,
        request_id=message.request_id,
        knowledge_references=message.knowledge_references,
        agent_steps=message.agent_steps,
        tool_calls=message.tool_calls,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


@router.get(
    "",
    response_model=MessageListResponse,
    summary="获取消息列表",
    description="分页获取指定会话的消息列表",
)
@limiter.limit(RateLimit.API)
async def list_messages(
    session: Annotated[AsyncSession, Depends(get_session)],
    request: StarletteRequest,
    tenant_id: TenantIdDep = None,
    session_id: str = Query(..., description="会话 ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(50, ge=1, le=100, description="每页数量"),
) -> MessageListResponse:
    """获取消息列表"""
    user_id = getattr(request.state, "user_id", None)

    service = MessageService(session)
    result = await service.list_messages(
        session_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
        page=page,
        size=size,
    )

    items = [_convert_to_response(msg) for msg in result.items]
    return MessageListResponse(
        items=items,
        total=result.total,
        page=result.page,
        size=result.size,
        pages=result.pages,
    )


@router.get(
    "/{message_id}",
    response_model=MessageResponse,
    summary="获取消息详情",
    description="根据 ID 获取消息的详细信息",
)
@limiter.limit(RateLimit.API)
async def get_message(
    session: Annotated[AsyncSession, Depends(get_session)],
    request: StarletteRequest,
    message_id: int,
    tenant_id: TenantIdDep = None,
) -> MessageResponse:
    """获取消息详情"""
    user_id = getattr(request.state, "user_id", None)

    service = MessageService(session)
    message = await service.get_message_or_404(
        message_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    return _convert_to_response(message)


@router.patch(
    "/{message_id}",
    response_model=MessageResponse,
    summary="编辑消息",
    description="编辑用户消息内容，可选择是否重新生成后续回复",
)
@limiter.limit(RateLimit.API)
async def update_message(
    session: Annotated[AsyncSession, Depends(get_session)],
    request: StarletteRequest,
    message_id: int,
    data: MessageUpdate,
    regenerate: MessageRegenerateRequest | None = None,
    tenant_id: TenantIdDep = None,
) -> MessageResponse:
    """编辑消息"""
    user_id = getattr(request.state, "user_id", None)

    service = MessageService(session)
    message = await service.update_message(
        message_id,
        data,
        regenerate_request=regenerate,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    return _convert_to_response(message)


@router.delete(
    "/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除消息",
    description="软删除指定的消息",
)
@limiter.limit(RateLimit.API)
async def delete_message(
    session: Annotated[AsyncSession, Depends(get_session)],
    request: StarletteRequest,
    message_id: int,
    tenant_id: TenantIdDep = None,
) -> None:
    """删除消息"""
    user_id = getattr(request.state, "user_id", None)

    service = MessageService(session)
    success = await service.delete_message(
        message_id,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="消息不存在",
        )

    logger.info("message_deleted_api", message_id=message_id)


@router.get(
    "/search",
    response_model=MessageSearchResponse,
    summary="搜索消息",
    description="在指定会话中搜索包含关键词的消息",
)
@limiter.limit(RateLimit.API)
async def search_messages(
    session: Annotated[AsyncSession, Depends(get_session)],
    request: StarletteRequest,
    tenant_id: TenantIdDep = None,
    q: str = Query(..., min_length=1, description="搜索关键词"),
    session_id: str = Query(..., description="会话 ID"),
    limit: int = Query(20, ge=1, le=100, description="限制数量"),
) -> MessageSearchResponse:
    """搜索消息"""
    user_id = getattr(request.state, "user_id", None)

    service = MessageService(session)
    messages = await service.search_messages(
        session_id,
        q,
        user_id=int(user_id) if user_id else None,
        tenant_id=tenant_id,
        limit=limit,
    )

    items = [_convert_to_response(msg) for msg in messages]
    return MessageSearchResponse(
        items=items,
        total=len(items),
        query=q,
    )
