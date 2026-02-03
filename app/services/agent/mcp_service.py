"""MCP 服务管理服务层

提供 MCP 服务配置的 CRUD 业务逻辑。
封装响应构建和权限检查逻辑，消除路由层的重复代码。
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.mcp.utils import load_mcp_tools, mcp_tool_to_dict
from app.infra.database import get_session
from app.models import MCPService, MCPServiceCreate, MCPServiceUpdate
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

    async def get_service_tools(
        self,
        service_id: int,
        tenant_id: int,
        timeout_seconds: int = 60,
    ) -> dict:
        """获取 MCP 服务提供的工具列表

        Args:
            service_id: 服务 ID
            tenant_id: 租户 ID
            timeout_seconds: 超时时间（秒）

        Returns:
            工具列表响应

        Raises:
            HTTPException: 服务不存在、无权限或连接失败时
        """
        service = await self._get_and_verify(service_id, tenant_id)

        if not service.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MCP 服务未启用，无法获取工具列表",
            )

        try:
            # 根据传输类型准备参数
            command = None
            args = None
            env = None
            url = None
            headers = None

            if service.transport_type == "stdio" and service.stdio_config:
                command = service.stdio_config.get("command")
                args = service.stdio_config.get("args")
                env = service.env_vars
            elif service.transport_type in ("sse", "http") and service.url:
                url = service.url
                headers = service.headers

            # 使用 load_mcp_tools 加载工具
            tools = await load_mcp_tools(
                server_type=service.transport_type,
                command=command,
                args=args,
                url=url,
                env=env,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )

            # 转换为字典格式
            tool_dicts = [mcp_tool_to_dict(t) for t in tools]

            logger.info(
                "mcp_service_tools_loaded",
                service_id=service_id,
                tenant_id=tenant_id,
                tool_count=len(tool_dicts),
            )

            return {
                "service_id": service_id,
                "service_name": service.name,
                "tools": tool_dicts,
                "total": len(tool_dicts),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "mcp_service_tools_load_failed",
                service_id=service_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"加载 MCP 工具失败: {str(e)}",
            ) from e

    async def validate_service_config(
        self,
        data: MCPServiceCreate,
        tenant_id: int,
        timeout_seconds: int = 30,
    ) -> dict:
        """验证 MCP 服务配置

        在创建服务前验证配置是否正确，尝试连接并获取工具列表。

        Args:
            data: 服务配置数据
            tenant_id: 租户 ID
            timeout_seconds: 超时时间（秒）

        Returns:
            验证结果，包含是否成功和工具列表预览

        Raises:
            HTTPException: 配置无效或连接失败时
        """
        try:
            # 根据传输类型准备参数
            command = None
            args = None
            env = None
            url = None
            headers = None

            if data.transport_type == "stdio" and data.stdio_config:
                command = data.stdio_config.get("command")
                args = data.stdio_config.get("args")
                env = data.env_vars
            elif data.transport_type in ("sse", "http") and data.url:
                url = data.url
                headers = data.headers

            # 尝试加载工具
            tools = await load_mcp_tools(
                server_type=data.transport_type,
                command=command,
                args=args,
                url=url,
                env=env,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )

            # 转换为字典格式（预览）
            tool_preview = [
                {"name": t.name, "description": t.description}
                for t in tools[:5]  # 只显示前 5 个工具作为预览
            ]

            logger.info(
                "mcp_service_config_validated",
                tenant_id=tenant_id,
                transport_type=data.transport_type,
                tool_count=len(tools),
            )

            return {
                "valid": True,
                "message": "MCP 服务配置验证成功",
                "tool_preview": tool_preview,
                "total_tools": len(tools),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "mcp_service_config_validation_failed",
                tenant_id=tenant_id,
                error=str(e),
            )
            return {
                "valid": False,
                "message": f"配置验证失败: {str(e)}",
                "tool_preview": [],
                "total_tools": 0,
            }


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
