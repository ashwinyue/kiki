"""MCP 服务管理服务层

提供 MCP 服务配置的 CRUD 业务逻辑。
封装响应构建和权限检查逻辑，消除路由层的重复代码。
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.models.agent import (
    MCPService,
    MCPServiceCreate,
    MCPServiceUpdate,
)
from app.observability.logging import get_logger
from app.repositories.mcp_service import MCPServiceRepository

logger = get_logger(__name__)


class McpServiceService:
    """MCP 服务管理服务

    封装 MCP 服务的业务逻辑，包括：
    - 创建服务配置
    - 查询和列表
    - 更新和删除
    - 租户权限验证
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._repository: MCPServiceRepository | None = None

    @property
    def repository(self) -> MCPServiceRepository:
        """获取仓储（延迟初始化）"""
        if self._repository is None:
            self._repository = MCPServiceRepository(self.session)
        return self._repository

    # ============== 响应构建方法 ==============

    def _to_dict(self, service: MCPService) -> dict:
        """转换为字典响应

        Args:
            service: MCP Service 模型

        Returns:
            字典格式的响应
        """
        return {
            "id": service.id,
            "tenant_id": service.tenant_id,
            "name": service.name,
            "description": service.description,
            "enabled": service.enabled,
            "transport_type": service.transport_type,
            "url": service.url,
            "headers": service.headers,
            "auth_config": service.auth_config,
            "advanced_config": service.advanced_config,
            "stdio_config": service.stdio_config,
            "env_vars": service.env_vars,
            "created_at": service.created_at.isoformat() if service.created_at else None,
        }

    # ============== 权限验证 ==============

    async def _verify_access(self, service: MCPService, tenant_id: int) -> MCPService:
        """验证租户对服务的访问权限

        Args:
            service: MCP Service 模型
            tenant_id: 租户 ID

        Returns:
            MCP Service 模型

        Raises:
            HTTPException: 服务不存在或无权限时
        """
        if service.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP 服务不存在",
            )

        if service.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问此服务",
            )

        return service

    async def _get_and_verify(
        self,
        service_id: int,
        tenant_id: int,
    ) -> MCPService:
        """获取服务并验证权限

        Args:
            service_id: 服务 ID
            tenant_id: 租户 ID

        Returns:
            MCP Service 模型

        Raises:
            HTTPException: 服务不存在或无权限时
        """
        service = await self.repository.get(service_id)

        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP 服务不存在",
            )

        return await self._verify_access(service, tenant_id)

    # ============== CRUD 操作 ==============

    async def list_services(
        self,
        tenant_id: int,
        include_disabled: bool = True,
    ) -> list[dict]:
        """列出租户的 MCP 服务

        Args:
            tenant_id: 租户 ID
            include_disabled: 是否包含禁用的服务

        Returns:
            服务列表（字典格式）
        """
        services = await self.repository.list_by_tenant(
            tenant_id=tenant_id,
            include_disabled=include_disabled,
        )

        return [self._to_dict(svc) for svc in services]

    async def get_service(
        self,
        service_id: int,
        tenant_id: int,
    ) -> dict:
        """获取 MCP 服务详情

        Args:
            service_id: 服务 ID
            tenant_id: 租户 ID

        Returns:
            服务详情（字典格式）

        Raises:
            HTTPException: 服务不存在或无权限时
        """
        service = await self._get_and_verify(service_id, tenant_id)
        return self._to_dict(service)

    async def create_service(
        self,
        data: MCPServiceCreate,
        tenant_id: int,
    ) -> dict:
        """创建 MCP 服务

        Args:
            data: 创建数据
            tenant_id: 租户 ID

        Returns:
            创建的服务（字典格式）
        """
        # 确保租户 ID 正确设置
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

        service = await self.repository.create_service(create_data)

        logger.info(
            "mcp_service_created",
            service_id=service.id,
            tenant_id=tenant_id,
            name=data.name,
        )

        return self._to_dict(service)

    async def update_service(
        self,
        service_id: int,
        tenant_id: int,
        data: MCPServiceUpdate,
    ) -> dict:
        """更新 MCP 服务

        Args:
            service_id: 服务 ID
            tenant_id: 租户 ID
            data: 更新数据

        Returns:
            更新后的服务（字典格式）

        Raises:
            HTTPException: 服务不存在或无权限时
        """
        service = await self._get_and_verify(service_id, tenant_id)

        # 更新字段
        updated_service = await self.repository.update_service(service, data)

        await self.session.commit()
        await self.session.refresh(updated_service)

        logger.info(
            "mcp_service_updated",
            service_id=service_id,
            tenant_id=tenant_id,
        )

        return self._to_dict(updated_service)

    async def delete_service(
        self,
        service_id: int,
        tenant_id: int,
    ) -> None:
        """删除 MCP 服务（软删除）

        Args:
            service_id: 服务 ID
            tenant_id: 租户 ID

        Raises:
            HTTPException: 服务不存在或无权限时
        """
        service = await self._get_and_verify(service_id, tenant_id)

        # 软删除：禁用并标记删除时间
        service.enabled = False
        service.deleted_at = datetime.now(UTC)

        await self.session.commit()

        logger.info(
            "mcp_service_deleted",
            service_id=service_id,
            tenant_id=tenant_id,
        )


# ============== 依赖注入工厂 ==============


def get_mcp_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> McpServiceService:
    """创建 MCP 服务管理服务实例

    Args:
        session: 数据库会话

    Returns:
        McpServiceService 实例
    """
    return McpServiceService(session)


# 向后兼容别名
get_mcp_service_service = get_mcp_service
