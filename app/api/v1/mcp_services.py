"""MCP 服务管理 API

提供 MCP 服务配置的创建、列表、查询、更新、删除等功能。
使用 Service 层处理业务逻辑，API 层仅负责请求/响应处理。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request as StarletteRequest

from app.rate_limit.limiter import RateLimit, limiter
from app.middleware import RequiredTenantIdDep
from app.models.agent import MCPServiceCreate, MCPServiceUpdate
from app.observability.logging import get_logger
from app.infra.database import get_session
from app.schemas.mcp_service import (
    MCPServiceListResponse,
    MCPServiceRequest,
    MCPServiceResponse,
)
from app.services.mcp_service_service import (
    McpServiceService,
    get_mcp_service_service,
)

router = APIRouter(prefix="/mcp-services", tags=["MCP Services"])
logger = get_logger(__name__)


# ============== CRUD 接口 ==============


@router.get(
    "",
    response_model=MCPServiceListResponse,
    summary="列出 MCP 服务",
    description="获取当前租户的所有 MCP 服务配置",
)
@limiter.limit(RateLimit.API)
async def list_mcp_services(
    request: StarletteRequest,
    tenant_id: RequiredTenantIdDep,
    service: Annotated[McpServiceService, Depends(get_mcp_service_service)],
    include_disabled: bool = Query(True, description="是否包含禁用服务"),
) -> MCPServiceListResponse:
    """列出租户的 MCP 服务"""
    items = await service.list_services(tenant_id, include_disabled)
    return MCPServiceListResponse(items=items, total=len(items))


@router.post(
    "",
    response_model=MCPServiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建 MCP 服务",
    description="创建新的 MCP 服务配置",
    responses={
        status.HTTP_201_CREATED: {"description": "MCP 服务创建成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
    },
)
@limiter.limit(RateLimit.API)
async def create_mcp_service(
    request: StarletteRequest,
    tenant_id: RequiredTenantIdDep,
    service: Annotated[McpServiceService, Depends(get_mcp_service_service)],
    data: MCPServiceRequest,
) -> dict:
    """创建 MCP 服务"""
    create_data = MCPServiceCreate(
        name=data.name,
        description=data.description,
        tenant_id=tenant_id,
        enabled=data.enabled,
        transport_type=data.transport_type,
        url=data.url,
        headers=data.headers,
        auth_config=data.auth_config,
        advanced_config=data.advanced_config,
        stdio_config=data.stdio_config,
        env_vars=data.env_vars,
    )
    return await service.create_service(create_data, tenant_id)


@router.get(
    "/{service_id}",
    response_model=MCPServiceResponse,
    summary="获取 MCP 服务详情",
    description="获取指定 MCP 服务的配置详情",
    responses={
        status.HTTP_200_OK: {"description": "成功返回服务详情"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "服务不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def get_mcp_service(
    request: StarletteRequest,
    tenant_id: RequiredTenantIdDep,
    service: Annotated[McpServiceService, Depends(get_mcp_service_service)],
    service_id: int,
) -> dict:
    """获取 MCP 服务详情"""
    return await service.get_service(service_id, tenant_id)


@router.patch(
    "/{service_id}",
    response_model=MCPServiceResponse,
    summary="更新 MCP 服务",
    description="更新 MCP 服务的配置",
    responses={
        status.HTTP_200_OK: {"description": "服务更新成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "服务不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def update_mcp_service(
    request: StarletteRequest,
    tenant_id: RequiredTenantIdDep,
    session: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[McpServiceService, Depends(get_mcp_service_service)],
    service_id: int,
    data: MCPServiceRequest,
) -> dict:
    """更新 MCP 服务"""
    update_data = MCPServiceUpdate(
        name=data.name,
        description=data.description,
        enabled=data.enabled,
        transport_type=data.transport_type,
        url=data.url,
        headers=data.headers,
        auth_config=data.auth_config,
        advanced_config=data.advanced_config,
        stdio_config=data.stdio_config,
        env_vars=data.env_vars,
    )
    return await service.update_service(service_id, tenant_id, update_data)


@router.delete(
    "/{service_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除 MCP 服务",
    description="删除指定的 MCP 服务（软删除）",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "服务删除成功"},
        status.HTTP_401_UNAUTHORIZED: {"description": "未认证"},
        status.HTTP_403_FORBIDDEN: {"description": "无权访问"},
        status.HTTP_404_NOT_FOUND: {"description": "服务不存在"},
    },
)
@limiter.limit(RateLimit.API)
async def delete_mcp_service(
    request: StarletteRequest,
    tenant_id: RequiredTenantIdDep,
    service: Annotated[McpServiceService, Depends(get_mcp_service_service)],
    service_id: int,
) -> None:
    """删除 MCP 服务"""
    await service.delete_service(service_id, tenant_id)
