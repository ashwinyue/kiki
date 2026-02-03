"""租户 KV 配置 API 路由

对齐 WeKnora99 租户配置管理 API
使用 FastAPI 标准依赖注入模式。
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.dependencies import get_session_dep
from app.middleware import TenantIdDep
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository
from app.schemas.response import ApiResponse, DataResponse

router = APIRouter(prefix="/tenants/kv", tags=["tenant-config"])
logger = get_logger(__name__)

# ============== 依赖类型别名 ==============

# 数据库会话依赖
DbDep = Annotated[AsyncSession, Depends(get_session_dep)]


class TenantConfigRepository(BaseRepository[dict]):
    """租户配置仓储（使用 JSONB 存储）"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, tenant_id: int, key: str) -> Any | None:
        """获取配置值"""
        from sqlalchemy import select

        from app.models.tenant import Tenant

        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant or not tenant.kv_config:
            return None

        return tenant.kv_config.get(key)

    async def set(self, tenant_id: int, key: str, value: Any) -> None:
        """设置配置值"""
        from sqlalchemy import select

        from app.models.tenant import Tenant

        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant:
            raise ValueError("Tenant not found")

        if tenant.kv_config is None:
            tenant.kv_config = {}

        tenant.kv_config[key] = value
        await self.session.commit()

    async def get_all(self, tenant_id: int) -> dict:
        """获取所有配置"""
        from sqlalchemy import select

        from app.models.tenant import Tenant

        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant:
            return {}

        return tenant.kv_config or {}


@router.get("/{key}")
async def get_tenant_config(
    key: str,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取配置值

    对齐 WeKnora99 GET /tenants/kv/{key}
    """
    repo = TenantConfigRepository(db)
    value = await repo.get(tenant_id, key)

    if value is None:
        raise HTTPException(status_code=404, detail="Config key not found")

    logger.info("tenant_config_get", tenant_id=tenant_id, key=key)

    return ApiResponse(success=True, data=value)


@router.put("/{key}")
async def set_tenant_config(
    key: str,
    value: Any,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新配置值

    对齐 WeKnora99 PUT /tenants/kv/{key}
    """
    repo = TenantConfigRepository(db)
    await repo.set(tenant_id, key, value)

    logger.info("tenant_config_set", tenant_id=tenant_id, key=key)

    return ApiResponse(success=True, message="Config updated")


@router.get("/agent-config", response_model=DataResponse[dict])
async def get_agent_config(
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取 Agent 配置

    对齐 WeKnora99 GET /tenants/kv/agent-config
    """
    repo = TenantConfigRepository(db)
    config = await repo.get(tenant_id, "agent_config")

    if not config:
        # 返回默认配置
        config = {
            "agent_mode": "quick-answer",
            "temperature": 0.7,
            "max_iterations": 10,
            "web_search_enabled": True,
        }

    return DataResponse(success=True, data=config)


@router.put("/agent-config")
async def set_agent_config(
    config: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新 Agent 配置

    对齐 WeKnora99 PUT /tenants/kv/agent-config
    """
    repo = TenantConfigRepository(db)
    await repo.set(tenant_id, "agent_config", config)

    logger.info(
        "agent_config_updated",
        tenant_id=tenant_id,
        config_keys=list(config.keys()) if config else [],
    )

    return ApiResponse(success=True, message="Agent config updated")


@router.get("/web-search-config", response_model=DataResponse[dict])
async def get_web_search_config(
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """获取网络搜索配置

    对齐 WeKnora99 GET /tenants/kv/web-search-config
    """
    repo = TenantConfigRepository(db)
    config = await repo.get(tenant_id, "web_search_config")

    if not config:
        # 返回默认配置
        config = {
            "enabled": True,
            "provider": "auto",
            "max_results": 5,
        }

    return DataResponse(success=True, data=config)


@router.put("/web-search-config")
async def set_web_search_config(
    config: dict,
    db: DbDep,
    tenant_id: TenantIdDep,
):
    """更新网络搜索配置

    对齐 WeKnora99 PUT /tenants/kv/web-search-config
    """
    repo = TenantConfigRepository(db)
    await repo.set(tenant_id, "web_search_config", config)

    logger.info(
        "web_search_config_updated",
        tenant_id=tenant_id,
        enabled=config.get("enabled", False),
    )

    return ApiResponse(success=True, message="Web search config updated")


__all__ = ["router"]
