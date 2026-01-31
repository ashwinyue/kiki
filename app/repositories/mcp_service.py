"""MCP 服务仓储

提供 MCP 服务配置的数据访问操作。
"""

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mcp_service import MCPService, MCPServiceCreate, MCPServiceUpdate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository

logger = get_logger(__name__)


class MCPServiceRepository(BaseRepository[MCPService]):
    """MCP 服务仓储类"""

    def __init__(self, session: AsyncSession):
        super().__init__(MCPService, session)

    async def create_service(self, data: MCPServiceCreate) -> MCPService:
        """创建 MCP 服务"""
        service = MCPService(**data.model_dump())
        return await self.create(service)

    async def list_enabled(
        self,
        tenant_id: int | None = None,
        include_global: bool = True,
    ) -> list[MCPService]:
        """列出启用的 MCP 服务（可按租户过滤）"""
        try:
            statement = select(MCPService).where(
                MCPService.enabled,
                MCPService.deleted_at.is_(None),
            )
            if tenant_id is not None:
                if include_global:
                    statement = statement.where(
                        or_(MCPService.tenant_id == tenant_id, MCPService.tenant_id.is_(None))
                    )
                else:
                    statement = statement.where(MCPService.tenant_id == tenant_id)
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("mcp_service_repository_list_enabled_failed", error=str(e))
            return []

    async def list_by_tenant(
        self,
        tenant_id: int,
        include_disabled: bool = True,
    ) -> list[MCPService]:
        """列出租户 MCP 服务（默认包含禁用）"""
        try:
            statement = select(MCPService).where(
                MCPService.tenant_id == tenant_id,
                MCPService.deleted_at.is_(None),
            )
            if not include_disabled:
                statement = statement.where(MCPService.enabled)
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("mcp_service_repository_list_by_tenant_failed", error=str(e))
            return []

    async def update_service(self, service: MCPService, data: MCPServiceUpdate) -> MCPService:
        """更新 MCP 服务"""
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(service, field, value)
        self.session.add(service)
        await self.session.flush()
        return service
