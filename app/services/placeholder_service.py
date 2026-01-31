"""占位符服务

提供占位符的 CRUD 操作和模板渲染功能。
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.placeholder import Placeholder
from app.observability.logging import get_logger
from app.repositories.placeholder import PlaceholderRepository
from app.schemas.agent import (
    PlaceholderCreate,
    PlaceholderPreviewRequest,
    PlaceholderPreviewResponse,
    PlaceholderSchema,
    PlaceholderUpdate,
)
from app.utils.template import preview_render

logger = get_logger(__name__)


class PlaceholderService:
    """占位符服务

    提供占位符的 CRUD 操作和模板渲染功能。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self.repository = PlaceholderRepository(session)

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
        result = await self.repository.list_by_tenant(
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
                description=p.description,
                default_value=p.default_value,
                variable_type=p.variable_type,
                validation_rule=p.validation_rule,
                is_required=p.is_required,
                is_enabled=p.is_enabled,
                agent_id=str(p.agent_id) if p.agent_id else None,
                category=p.category,
                display_order=p.display_order,
                created_at=p.created_at.isoformat() if p.created_at else "",
            )
            for p in result.items
        ]

        return items, result.total

    async def get_placeholder(self, placeholder_id: str) -> PlaceholderSchema | None:
        """获取占位符详情

        Args:
            placeholder_id: 占位符 ID

        Returns:
            占位符详情，不存在返回 None
        """
        placeholder = await self.repository.get(placeholder_id)
        if placeholder is None or placeholder.deleted_at is not None:
            return None

        return PlaceholderSchema(
            id=str(placeholder.id),
            name=placeholder.name,
            description=placeholder.description,
            default_value=placeholder.default_value,
            variable_type=placeholder.variable_type,
            validation_rule=placeholder.validation_rule,
            is_required=placeholder.is_required,
            is_enabled=placeholder.is_enabled,
            agent_id=str(placeholder.agent_id) if placeholder.agent_id else None,
            category=placeholder.category,
            display_order=placeholder.display_order,
            created_at=placeholder.created_at.isoformat() if placeholder.created_at else "",
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

        Raises:
            HTTPException: 名称已存在
        """
        # 检查名称是否已存在
        existing = await self.repository.get_by_name(
            data.name,
            tenant_id=tenant_id,
            agent_id=data.agent_id,
        )
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"占位符 '{data.name}' 已存在",
            )

        # 构建数据
        placeholder_data = {
            "id": str(uuid4()),
            "name": data.name,
            "description": data.description,
            "default_value": data.default_value,
            "variable_type": data.variable_type,
            "validation_rule": data.validation_rule,
            "is_required": data.is_required,
            "is_enabled": data.is_enabled,
            "tenant_id": tenant_id,
            "agent_id": data.agent_id,
            "category": data.category,
            "display_order": data.display_order,
            "created_by": created_by,
        }

        placeholder = await self.repository.create_with_metadata(placeholder_data)

        return PlaceholderSchema(
            id=str(placeholder.id),
            name=placeholder.name,
            description=placeholder.description,
            default_value=placeholder.default_value,
            variable_type=placeholder.variable_type,
            validation_rule=placeholder.validation_rule,
            is_required=placeholder.is_required,
            is_enabled=placeholder.is_enabled,
            agent_id=str(placeholder.agent_id) if placeholder.agent_id else None,
            category=placeholder.category,
            display_order=placeholder.display_order,
            created_at=placeholder.created_at.isoformat() if placeholder.created_at else "",
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
            tenant_id: 租户 ID（用于权限检查）

        Returns:
            更新后的占位符，不存在返回 None

        Raises:
            HTTPException: 名称冲突
        """
        placeholder = await self.repository.get(placeholder_id)
        if placeholder is None or placeholder.deleted_at is not None:
            return None

        # 租户权限检查
        if tenant_id is not None and placeholder.tenant_id != tenant_id:
            return None

        # 如果修改名称，检查新名称是否已存在
        if data.name and data.name != placeholder.name:
            existing = await self.repository.get_by_name(
                data.name,
                tenant_id=placeholder.tenant_id,
                agent_id=placeholder.agent_id,
            )
            if existing is not None and existing.id != placeholder_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"占位符 '{data.name}' 已存在",
                )

        # 构建更新数据
        update_data = data.model_dump(exclude_unset=True)

        updated = await self.repository.update_placeholder(placeholder_id, update_data)
        if updated is None:
            return None

        return PlaceholderSchema(
            id=str(updated.id),
            name=updated.name,
            description=updated.description,
            default_value=updated.default_value,
            variable_type=updated.variable_type,
            validation_rule=updated.validation_rule,
            is_required=updated.is_required,
            is_enabled=updated.is_enabled,
            agent_id=str(updated.agent_id) if updated.agent_id else None,
            category=updated.category,
            display_order=updated.display_order,
            created_at=updated.created_at.isoformat() if updated.created_at else "",
        )

    async def delete_placeholder(
        self,
        placeholder_id: str,
        tenant_id: int | None = None,
    ) -> bool:
        """删除占位符

        Args:
            placeholder_id: 占位符 ID
            tenant_id: 租户 ID（用于权限检查）

        Returns:
            是否删除成功
        """
        placeholder = await self.repository.get(placeholder_id)
        if placeholder is None or placeholder.deleted_at is not None:
            return False

        # 租户权限检查
        if tenant_id is not None and placeholder.tenant_id != tenant_id:
            return False

        return await self.repository.soft_delete(placeholder_id)

    async def render_template(
        self,
        template: str,
        values: dict[str, Any],
        *,
        agent_id: str | None = None,
        tenant_id: int | None = None,
    ) -> str:
        """渲染模板

        Args:
            template: 模板字符串
            values: 变量值
            agent_id: Agent ID（用于获取关联占位符）
            tenant_id: 租户 ID

        Returns:
            渲染后的字符串
        """
        # 获取关联的占位符定义
        placeholders = []
        if agent_id:
            placeholder_list = await self.repository.list_by_agent(agent_id, is_enabled=True)
            placeholders = [
                {
                    "name": p.name,
                    "variable_type": p.variable_type,
                    "is_required": p.is_required,
                    "default_value": p.default_value,
                    "validation_rule": p.validation_rule,
                }
                for p in placeholder_list
                if p.is_enabled
            ]

        # 获取全局占位符
        if tenant_id:
            from app.repositories.base import PaginationParams

            global_result = await self.repository.list_by_tenant(
                tenant_id,
                agent_id=None,
                is_enabled=True,
                params=PaginationParams(page=1, size=1000),
            )
            for p in global_result.items:
                if p.is_enabled and not any(
                    gp["name"] == p.name for gp in placeholders
                ):
                    placeholders.append(
                        {
                            "name": p.name,
                            "variable_type": p.variable_type,
                            "is_required": p.is_required,
                            "default_value": p.default_value,
                            "validation_rule": p.validation_rule,
                        }
                    )

        # 使用 preview_render 进行渲染
        result = preview_render(template, values, placeholders)

        # 如果有验证错误，记录日志但仍然返回渲染结果
        if result["validation_errors"]:
            logger.warning(
                "template_render_validation_warnings",
                errors=result["validation_errors"],
            )

        return result["rendered"]

    async def preview(
        self,
        request: PlaceholderPreviewRequest,
        tenant_id: int | None = None,
    ) -> PlaceholderPreviewResponse:
        """预览模板渲染

        Args:
            request: 预览请求
            tenant_id: 租户 ID

        Returns:
            预览响应
        """
        # 获取关联的占位符定义
        placeholders = []
        if request.agent_id:
            placeholder_list = await self.repository.list_by_agent(
                request.agent_id,
                is_enabled=True,
            )
            placeholders = [
                {
                    "name": p.name,
                    "variable_type": p.variable_type,
                    "is_required": p.is_required,
                    "default_value": p.default_value,
                    "validation_rule": p.validation_rule,
                }
                for p in placeholder_list
                if p.is_enabled
            ]

        # 获取全局占位符
        if tenant_id:
            from app.repositories.base import PaginationParams

            global_result = await self.repository.list_by_tenant(
                tenant_id,
                agent_id=None,
                is_enabled=True,
                params=PaginationParams(page=1, size=1000),
            )
            for p in global_result.items:
                if p.is_enabled and not any(
                    gp["name"] == p.name for gp in placeholders
                ):
                    placeholders.append(
                        {
                            "name": p.name,
                            "variable_type": p.variable_type,
                            "is_required": p.is_required,
                            "default_value": p.default_value,
                            "validation_rule": p.validation_rule,
                        }
                    )

        # 使用 preview_render 进行预览
        result = preview_render(request.template, request.values, placeholders)

        return PlaceholderPreviewResponse(
            rendered=result["rendered"],
            missing_variables=result["missing_variables"],
            validation_errors=result["validation_errors"],
            used_placeholders=result["used_placeholders"],
        )

    async def get_placeholders_for_agent(
        self,
        agent_id: str,
        tenant_id: int,
    ) -> list[PlaceholderSchema]:
        """获取 Agent 的所有占位符（包括全局占位符）

        Args:
            agent_id: Agent ID
            tenant_id: 租户 ID

        Returns:
            占位符列表
        """
        # 获取 Agent 专属占位符
        agent_placeholders = await self.repository.list_by_agent(agent_id)
        agent_ids = {p.id for p in agent_placeholders}

        # 获取全局占位符
        from app.repositories.base import PaginationParams

        global_result = await self.repository.list_by_tenant(
            tenant_id,
            agent_id=None,
            params=PaginationParams(page=1, size=1000),
        )

        # 合并结果（Agent 占位符优先）
        all_placeholders = list(agent_placeholders)
        for p in global_result.items:
            if p.id not in agent_ids:
                all_placeholders.append(p)

        # 按显示顺序排序
        all_placeholders.sort(key=lambda p: (p.display_order, p.created_at or datetime.min))

        return [
            PlaceholderSchema(
                id=str(p.id),
                name=p.name,
                description=p.description,
                default_value=p.default_value,
                variable_type=p.variable_type,
                validation_rule=p.validation_rule,
                is_required=p.is_required,
                is_enabled=p.is_enabled,
                agent_id=str(p.agent_id) if p.agent_id else None,
                category=p.category,
                display_order=p.display_order,
                created_at=p.created_at.isoformat() if p.created_at else "",
            )
            for p in all_placeholders
        ]
