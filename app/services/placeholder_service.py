"""占位符服务

提供占位符的业务逻辑层。
"""

from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateError
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.repositories.placeholder import PlaceholderRepository
from app.schemas.agent import (
    PlaceholderCreate,
    PlaceholderPreviewRequest,
    PlaceholderPreviewResponse,
    PlaceholderSchema,
    PlaceholderUpdate,
)
from app.utils.template import render_template

logger = get_logger(__name__)


class PlaceholderService:
    """占位符服务

    提供占位符的业务逻辑处理。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化占位符服务

        Args:
            session: 异步数据库会话
        """
        self.session = session
        self._repo = PlaceholderRepository(session)

    async def list_placeholders(
        self,
        tenant_id: int,
        *,
        agent_id: str | None = None,
        category: str | None = None,
        is_enabled: bool | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[PlaceholderSchema], int]:
        """列出占位符

        Args:
            tenant_id: 租户 ID
            agent_id: 过滤 Agent ID
            category: 过滤分类
            is_enabled: 过滤是否启用
            page: 页码
            size: 每页数量

        Returns:
            (占位符列表, 总数)
        """
        from app.repositories.base import PaginationParams

        params = PaginationParams(page=page, size=size)
        result = await self._repo.list_by_tenant(
            tenant_id,
            agent_id=agent_id,
            category=category,
            is_enabled=is_enabled,
            params=params,
        )

        items = [
            PlaceholderSchema(
                id=str(p.id),
                name=p.name,
                key=p.key,
                value=p.value,
                description=p.description,
                category=p.category,
                agent_id=str(p.agent_id) if p.agent_id else None,
                is_enabled=p.is_enabled,
                display_order=p.display_order,
                created_at=p.created_at.isoformat() if p.created_at else None,
                updated_at=p.updated_at.isoformat() if p.updated_at else None,
            )
            for p in result.items
        ]

        return items, result.total

    async def get_placeholder(self, placeholder_id: str) -> PlaceholderSchema | None:
        """获取占位符详情

        Args:
            placeholder_id: 占位符 ID

        Returns:
            占位符详情或 None
        """
        placeholder = await self._repo.get(placeholder_id)
        if placeholder is None:
            return None

        return PlaceholderSchema(
            id=str(placeholder.id),
            name=placeholder.name,
            key=placeholder.key,
            value=placeholder.value,
            description=placeholder.description,
            category=placeholder.category,
            agent_id=str(placeholder.agent_id) if placeholder.agent_id else None,
            is_enabled=placeholder.is_enabled,
            display_order=placeholder.display_order,
            created_at=placeholder.created_at.isoformat() if placeholder.created_at else None,
            updated_at=placeholder.updated_at.isoformat() if placeholder.updated_at else None,
        )

    async def create_placeholder(
        self,
        data: PlaceholderCreate,
        tenant_id: int,
        created_by: str | None = None,
    ) -> PlaceholderSchema:
        """创建占位符

        Args:
            data: 创建数据
            tenant_id: 租户 ID
            created_by: 创建人 ID

        Returns:
            创建的占位符
        """
        from uuid import uuid4

        placeholder_data = {
            "id": str(uuid4()),
            "name": data.name,
            "key": data.key,
            "value": data.value,
            "description": data.description,
            "category": data.category,
            "tenant_id": tenant_id,
            "agent_id": data.agent_id,
            "is_enabled": data.is_enabled,
            "display_order": data.display_order,
            "created_by": created_by,
        }

        placeholder = await self._repo.create_with_metadata(placeholder_data)

        return PlaceholderSchema(
            id=str(placeholder.id),
            name=placeholder.name,
            key=placeholder.key,
            value=placeholder.value,
            description=placeholder.description,
            category=placeholder.category,
            agent_id=str(placeholder.agent_id) if placeholder.agent_id else None,
            is_enabled=placeholder.is_enabled,
            display_order=placeholder.display_order,
            created_at=placeholder.created_at.isoformat() if placeholder.created_at else None,
            updated_at=placeholder.updated_at.isoformat() if placeholder.updated_at else None,
        )

    async def update_placeholder(
        self,
        placeholder_id: str,
        data: PlaceholderUpdate,
        tenant_id: int | None = None,
    ) -> PlaceholderSchema | None:
        """更新占位符

        Args:
            placeholder_id: 占位符 ID
            data: 更新数据
            tenant_id: 租户 ID（用于权限验证）

        Returns:
            更新后的占位符或 None
        """
        update_data = data.model_dump(exclude_unset=True)
        placeholder = await self._repo.update_placeholder(placeholder_id, update_data)

        if placeholder is None:
            return None

        return PlaceholderSchema(
            id=str(placeholder.id),
            name=placeholder.name,
            key=placeholder.key,
            value=placeholder.value,
            description=placeholder.description,
            category=placeholder.category,
            agent_id=str(placeholder.agent_id) if placeholder.agent_id else None,
            is_enabled=placeholder.is_enabled,
            display_order=placeholder.display_order,
            created_at=placeholder.created_at.isoformat() if placeholder.created_at else None,
            updated_at=placeholder.updated_at.isoformat() if placeholder.updated_at else None,
        )

    async def delete_placeholder(
        self,
        placeholder_id: str,
        tenant_id: int | None = None,
    ) -> bool:
        """删除占位符

        Args:
            placeholder_id: 占位符 ID
            tenant_id: 租户 ID（用于权限验证）

        Returns:
            是否删除成功
        """
        return await self._repo.soft_delete(placeholder_id)

    async def get_placeholders_for_agent(
        self,
        agent_id: str,
        tenant_id: int,
    ) -> list[PlaceholderSchema]:
        """获取 Agent 的占位符

        Args:
            agent_id: Agent ID
            tenant_id: 租户 ID

        Returns:
            占位符列表
        """
        placeholders = await self._repo.list_by_agent(agent_id)

        return [
            PlaceholderSchema(
                id=str(p.id),
                name=p.name,
                key=p.key,
                value=p.value,
                description=p.description,
                category=p.category,
                agent_id=str(p.agent_id) if p.agent_id else None,
                is_enabled=p.is_enabled,
                display_order=p.display_order,
                created_at=p.created_at.isoformat() if p.created_at else None,
                updated_at=p.updated_at.isoformat() if p.updated_at else None,
            )
            for p in placeholders
        ]

    async def preview(
        self,
        data: PlaceholderPreviewRequest,
        tenant_id: int | None = None,
    ) -> PlaceholderPreviewResponse:
        """预览占位符替换

        Args:
            data: 预览请求数据
            tenant_id: 租户 ID

        Returns:
            预览响应
        """
        try:
            rendered = await render_template(
                data.template,
                data.variables,
                agent_id=data.agent_id,
                tenant_id=tenant_id,
            )

            return PlaceholderPreviewResponse(
                original=data.template,
                rendered=rendered,
                success=True,
                error=None,
            )
        except TemplateError as e:
            return PlaceholderPreviewResponse(
                original=data.template,
                rendered=data.template,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error("placeholder_preview_failed", error=str(e))
            return PlaceholderPreviewResponse(
                original=data.template,
                rendered=data.template,
                success=False,
                error=f"预览失败: {str(e)}",
            )
