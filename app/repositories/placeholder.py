"""占位符仓储模块

提供占位符的异步数据访问层。
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.placeholder import Placeholder
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class PlaceholderRepository(BaseRepository[Placeholder]):
    """占位符仓储

    提供占位符的异步 CRUD 操作。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化占位符仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(Placeholder, session)

    async def create_with_metadata(
        self,
        data: dict[str, Any],
    ) -> Placeholder:
        """创建占位符

        Args:
            data: 占位符创建数据

        Returns:
            创建的占位符
        """
        placeholder = Placeholder(**data)
        self.session.add(placeholder)
        await self.session.flush()

        await self.session.commit()
        await self.session.refresh(placeholder)

        logger.info(
            "placeholder_created",
            placeholder_id=placeholder.id,
            name=placeholder.name,
            agent_id=placeholder.agent_id,
        )
        return placeholder

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        agent_id: str | None = None,
        category: str | None = None,
        is_enabled: bool | None = None,
        params: PaginationParams | None = None,
    ) -> PaginatedResult[Placeholder]:
        """按租户分页列出占位符

        Args:
            tenant_id: 租户 ID
            agent_id: 过滤 Agent ID
            category: 过滤分类
            is_enabled: 过滤是否启用
            params: 分页参数

        Returns:
            分页结果
        """
        try:
            statement = select(Placeholder).where(
                Placeholder.tenant_id == tenant_id,
            )

            # 排除已删除的
            statement = statement.where(Placeholder.deleted_at.is_(None))

            # 过滤条件
            if agent_id is not None:
                # global (agent_id 为 null) 或特定 agent 的
                statement = statement.where(
                    (Placeholder.agent_id == agent_id) | (Placeholder.agent_id.is_(None))
                )
            if category is not None:
                statement = statement.where(Placeholder.category == category)
            if is_enabled is not None:
                statement = statement.where(Placeholder.is_enabled == is_enabled)

            # 按显示顺序和创建时间排序
            statement = statement.order_by(
                Placeholder.display_order.asc(),
                Placeholder.created_at.desc(),
            )

            # 获取总数
            from sqlalchemy import func

            count_stmt = select(func.count()).select_from(statement.subquery())
            total_result = await self.session.execute(count_stmt)
            total = total_result.scalar() or 0

            # 分页
            if params:
                statement = statement.offset(params.offset).limit(params.limit)
            else:
                params = PaginationParams(page=1, size=100)
                statement = statement.offset(params.offset).limit(params.limit)

            items_result = await self.session.execute(statement)
            items = list(items_result.scalars().all())

            return PaginatedResult.create(items, total, params)

        except Exception as e:
            logger.error(
                "placeholder_repository_list_failed",
                tenant_id=tenant_id,
                error=str(e),
            )
            return PaginatedResult.create([], 0, params or PaginationParams())

    async def list_by_agent(
        self,
        agent_id: str,
        *,
        is_enabled: bool | None = None,
    ) -> list[Placeholder]:
        """按 Agent 列出占位符

        Args:
            agent_id: Agent ID
            is_enabled: 过滤是否启用

        Returns:
            占位符列表
        """
        try:
            statement = select(Placeholder).where(
                (Placeholder.agent_id == agent_id) | (Placeholder.agent_id.is_(None)),
            )

            # 排除已删除的
            statement = statement.where(Placeholder.deleted_at.is_(None))

            if is_enabled is not None:
                statement = statement.where(Placeholder.is_enabled == is_enabled)

            statement = statement.order_by(
                Placeholder.display_order.asc(),
                Placeholder.created_at.desc(),
            )

            result = await self.session.execute(statement)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "placeholder_repository_list_by_agent_failed",
                agent_id=agent_id,
                error=str(e),
            )
            return []

    async def get_by_name(
        self,
        name: str,
        *,
        tenant_id: int | None = None,
        agent_id: str | None = None,
    ) -> Placeholder | None:
        """根据名称获取占位符

        Args:
            name: 占位符名称
            tenant_id: 租户 ID
            agent_id: Agent ID

        Returns:
            占位符实例或 None
        """
        try:
            statement = select(Placeholder).where(Placeholder.name == name)

            # 排除已删除的
            statement = statement.where(Placeholder.deleted_at.is_(None))

            if tenant_id is not None:
                statement = statement.where(Placeholder.tenant_id == tenant_id)

            if agent_id is not None:
                statement = statement.where(
                    (Placeholder.agent_id == agent_id) | (Placeholder.agent_id.is_(None))
                )

            result = await self.session.execute(statement)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(
                "placeholder_repository_get_by_name_failed",
                name=name,
                error=str(e),
            )
            return None

    async def soft_delete(self, placeholder_id: str) -> bool:
        """软删除占位符

        Args:
            placeholder_id: 占位符 ID

        Returns:
            是否删除成功
        """
        try:
            from datetime import UTC, datetime

            placeholder = await self.get(placeholder_id)
            if placeholder is None:
                return False

            placeholder.deleted_at = datetime.now(UTC)
            await self.session.commit()

            logger.info("placeholder_soft_deleted", placeholder_id=placeholder_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "placeholder_repository_delete_failed",
                placeholder_id=placeholder_id,
                error=str(e),
            )
            return False

    async def update_placeholder(
        self,
        placeholder_id: str,
        data: dict[str, Any],
    ) -> Placeholder | None:
        """更新占位符

        Args:
            placeholder_id: 占位符 ID
            data: 更新数据

        Returns:
            更新后的占位符或 None
        """
        try:
            placeholder = await self.get(placeholder_id)
            if placeholder is None:
                return None

            # 更新字段
            for field, value in data.items():
                if hasattr(placeholder, field):
                    setattr(placeholder, field, value)

            await self.session.commit()
            await self.session.refresh(placeholder)

            logger.info("placeholder_updated", placeholder_id=placeholder.id)
            return placeholder

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "placeholder_repository_update_failed",
                placeholder_id=placeholder_id,
                error=str(e),
            )
            raise
