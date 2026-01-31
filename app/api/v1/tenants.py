"""租户管理 API"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth.middleware import (
    require_tenant,
    TenantIdDep,
)
from app.models.database import Tenant, TenantCreate, TenantPublic, TenantUpdate
from app.services.tenant import TenantService
from app.tools.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/tenants", tags=["tenants"])


# ============== 响应模型 ==============


class TenantListResponse(BaseModel):
    """租户列表响应"""

    items: list[TenantPublic]
    total: int
    page: int = 1
    size: int = 20


class ApiKeyResponse(BaseModel):
    """API Key 响应"""

    api_key: str
    tenant_id: int


# ============== 管理端点 ==============


@router.post("", response_model=TenantPublic, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    data: TenantCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    _current_tenant_id: Annotated[int, Depends(require_tenant)] = None,
):
    """创建租户（需要认证）"""
    service = TenantService(session)

    # 检查名称是否重复
    existing = await service.get_tenant_by_name(data.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"租户名称 '{data.name}' 已存在",
        )

    return await service.create_tenant(data)


@router.get("", response_model=TenantListResponse)
async def list_tenants(
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
    status: str | None = None,
    keyword: str | None = None,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    """列出租户（支持分页和搜索）"""
    service = TenantService(session)
    tenant = await service.get_tenant_by_id(current_tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    return TenantListResponse(items=[tenant], total=1, page=1, size=1)


@router.get("/{tenant_id}", response_model=TenantPublic)
async def get_tenant(
    tenant_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """获取租户详情"""
    if tenant_id != current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问其他租户",
        )
    service = TenantService(session)
    tenant = await service.get_tenant_by_id(tenant_id)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    # TODO: 添加跨租户访问权限检查
    # 如果当前已认证，检查是否有权限访问目标租户
    return tenant


@router.patch("/{tenant_id}", response_model=TenantPublic)
async def update_tenant(
    tenant_id: int,
    data: TenantUpdate,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """更新租户"""
    if tenant_id != current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问其他租户",
        )
    service = TenantService(session)
    tenant = await service.update_tenant(tenant_id, data)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    return tenant


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(
    tenant_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """删除租户（软删除）"""
    if tenant_id != current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问其他租户",
        )
    service = TenantService(session)
    success = await service.delete_tenant(tenant_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )


@router.post(
    "/{tenant_id}/rotate-api-key",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_200_OK,
)
async def rotate_api_key(
    tenant_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """轮换 API Key

    生成新的 API Key，旧的立即失效。
    """
    if tenant_id != current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问其他租户",
        )

    service = TenantService(session)
    new_api_key = await service.rotate_api_key(tenant_id)

    if not new_api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    return ApiKeyResponse(api_key=new_api_key, tenant_id=tenant_id)


# ============== 租户自身操作 ==============


@router.get("/me/config", response_model=dict)
async def get_my_config(
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """获取当前租户配置"""
    service = TenantService(session)
    tenant = await service.get_tenant_by_id(current_tenant_id)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    return tenant.config or {}


@router.patch("/me/config", response_model=dict)
async def update_my_config(
    config_update: dict,
    session: Annotated[AsyncSession, Depends(get_session)],
    current_tenant_id: Annotated[int, Depends(require_tenant)],
):
    """更新当前租户配置"""
    service = TenantService(session)

    # 获取当前租户
    tenant = await service.get_tenant_by_id(current_tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="租户不存在",
        )

    # 合并配置
    current_config = tenant.config or {}
    current_config.update(config_update)

    # 更新
    await service.update_tenant(current_tenant_id, TenantUpdate(config=current_config))

    return current_config
