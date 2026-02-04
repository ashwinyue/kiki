"""API Key 管理 API

提供 API Key 的创建、列表、查询、更新、删除等功能。
使用 Service 层处理业务逻辑，API 层仅负责请求/响应处理。
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from starlette.requests import Request as StarletteRequest

from app.api.v1.auth import get_current_user_id as get_current_user_id_int
from app.models.api_key import (
    ApiKeyCreate,
    ApiKeyRead,
    ApiKeyResponse,
    ApiKeyStatus,
    ApiKeyType,
    ApiKeyUpdate,
    ApiKeyVerifyResponse,
)
from app.observability.logging import get_logger
from app.rate_limit.limiter import RateLimit, limiter
from app.services.agent.api_key_management_service import (
    ApiKeyManagementService,
    get_api_key_management_service,
)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建 API Key",
    description="创建一个新的 API Key，返回完整的 Key（仅此一次）",
    responses={
        status.HTTP_201_CREATED: {"description": "API Key 创建成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "请求过于频繁"},
    },
)
@limiter.limit(RateLimit.API)
async def create_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    data: ApiKeyCreate,
) -> ApiKeyResponse:
    """创建 API Key"""
    return await service.create_api_key(data, user_id)


@router.get(
    "",
    response_model=list[ApiKeyRead],
    summary="列出 API Key",
    description="列出当前用户的所有 API Key",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 API Key 列表"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def list_api_keys(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    key_type: ApiKeyType | None = None,
    status_filter: ApiKeyStatus | None = None,
) -> list[ApiKeyRead]:
    """列出用户的 API Key"""
    return await service.list_api_keys(user_id, status_filter, key_type)


@router.get(
    "/{api_key_id}",
    response_model=ApiKeyRead,
    summary="获取 API Key 详情",
    description="获取指定 API Key 的详细信息",
    responses={
        status.HTTP_200_OK: {"description": "成功返回 API Key 详情"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "API Key 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def get_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    api_key_id: int,
) -> ApiKeyRead:
    """获取 API Key 详情"""
    return await service.get_api_key(api_key_id, user_id)


@router.patch(
    "/{api_key_id}",
    response_model=ApiKeyRead,
    summary="更新 API Key",
    description="更新 API Key 的名称、状态或权限范围",
    responses={
        status.HTTP_200_OK: {"description": "API Key 更新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "API Key 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def update_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    api_key_id: int,
    *,
    data: ApiKeyUpdate,
) -> ApiKeyRead:
    """更新 API Key"""
    return await service.update_api_key(api_key_id, user_id, data)


@router.delete(
    "/{api_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除 API Key",
    description="软删除指定的 API Key",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "API Key 删除成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "API Key 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def delete_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    api_key_id: int,
) -> None:
    """删除 API Key"""
    await service.delete_api_key(api_key_id, user_id)


@router.post(
    "/{api_key_id}/revoke",
    response_model=ApiKeyRead,
    summary="吊销 API Key",
    description="立即吊销指定的 API Key",
    responses={
        status.HTTP_200_OK: {"description": "API Key 吊销成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "API Key 不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def revoke_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    api_key_id: int,
) -> ApiKeyRead:
    """吊销 API Key"""
    return await service.revoke_api_key(api_key_id, user_id)


# ============== 验证和统计接口 ==============


@router.post(
    "/verify",
    response_model=ApiKeyVerifyResponse,
    summary="验证 API Key",
    description="验证 API Key 是否有效（用于调试）",
)
@limiter.limit(RateLimit.API)
async def verify_api_key_endpoint(
    request: StarletteRequest,
    api_key: Annotated[Any, Depends(lambda: None)],  # 保持原有接口
) -> ApiKeyVerifyResponse:
    """验证 API Key"""
    # 保持原有验证逻辑

    # 实际验证由 require_api_key 处理
    # 这里只是为了保持 API 兼容性
    raise NotImplementedError("使用 require_api_key 依赖进行验证")


@router.get(
    "/stats/me",
    summary="获取 API Key 统计",
    description="获取当前用户的 API Key 统计信息",
)
@limiter.limit(RateLimit.API)
async def get_api_key_stats(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
) -> dict[str, Any]:
    """获取 API Key 统计"""
    return await service.get_stats(user_id)


# ============== MCP 专用接口 ==============


@router.post(
    "/mcp/create",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建 MCP 专用 API Key",
    description="创建一个用于 MCP 服务器的专用 API Key",
)
@limiter.limit(RateLimit.API)
async def create_mcp_api_key(
    request: StarletteRequest,
    service: Annotated[ApiKeyManagementService, Depends(get_api_key_management_service)],
    user_id: Annotated[int, Depends(get_current_user_id_int)],
    name: str,
    expires_in_days: int | None = None,
) -> ApiKeyResponse:
    """创建 MCP 专用 API Key"""
    return await service.create_mcp_api_key(user_id, name, expires_in_days)
