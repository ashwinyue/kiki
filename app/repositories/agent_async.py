"""Agent 异步仓储模块

提供 Agent 配置的异步数据访问层，兼容 BaseRepository 模式。
对齐 DeerFlow 和 WeKnora99 设计。
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_config import AgentConfig
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)

# 向后兼容别名
Agent = AgentConfig
CustomAgent = AgentConfig  # 向后兼容


class AgentRepositoryAsync(BaseRepository[Agent]):
    """Agent 异步仓储

    提供 Agent 配置的异步 CRUD 操作。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化 Agent 仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(Agent, session)

    async def create_with_tools(self, data: dict[str, Any]) -> Agent:
        """创建 Agent

        Args:
            data: Agent 创建数据

        Returns:
            创建的 Agent
        """
        agent = Agent(**data)
        self.session.add(agent)
        await self.session.flush()

        await self.session.commit()
        await self.session.refresh(agent)

        logger.info("agent_created", agent_id=agent.id, name=agent.name)
        return agent

    async def get_by_name(self, name: str) -> Agent | None:
        """根据名称获取 Agent

        Args:
            name: Agent 名称

        Returns:
            Agent 实例或 None
        """
        try:
            statement = select(Agent).where(Agent.name == name)
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("agent_repository_get_by_name_failed", name=name, error=str(e))
            return None

    async def list_by_tenant(
        self,
        tenant_id: int | None = None,
        params: PaginationParams | None = None,
    ) -> PaginatedResult[Agent]:
        """按租户分页列出 Agent

        Args:
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        try:
            statement = select(Agent)

            if tenant_id is not None:
                statement = statement.where(Agent.tenant_id == tenant_id)

            # 排除已删除的
            statement = statement.where(Agent.deleted_at.is_(None))

            # 按创建时间倒序
            statement = statement.order_by(desc(Agent.created_at))

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
            logger.error("agent_repository_list_failed", tenant_id=tenant_id, error=str(e))
            return PaginatedResult.create([], 0, params or PaginationParams())

    async def update_agent(
        self,
        agent_id: str,
        data: dict[str, Any],
    ) -> Agent | None:
        """更新 Agent

        Args:
            agent_id: Agent ID
            data: 更新数据

        Returns:
            更新后的 Agent 或 None
        """
        try:
            agent = await self.get(agent_id)
            if agent is None:
                return None

            # 更新字段
            for field, value in data.items():
                if hasattr(agent, field):
                    setattr(agent, field, value)

            await self.session.commit()
            await self.session.refresh(agent)

            logger.info("agent_updated", agent_id=agent.id)
            return agent

        except Exception as e:
            await self.session.rollback()
            logger.error("agent_repository_update_failed", agent_id=agent_id, error=str(e))
            raise

    async def soft_delete(self, agent_id: str) -> bool:
        """软删除 Agent

        Args:
            agent_id: Agent ID

        Returns:
            是否删除成功
        """
        try:
            agent = await self.get(agent_id)
            if agent is None:
                return False

            agent.deleted_at = datetime.now(UTC)
            await self.session.commit()

            logger.info("agent_soft_deleted", agent_id=agent_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("agent_repository_delete_failed", agent_id=agent_id, error=str(e))
            return False

    async def copy(
        self,
        source_agent_id: str,
        new_name: str,
        new_description: str | None = None,
        new_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> Agent | None:
        """复制 Agent

        Args:
            source_agent_id: 源 Agent ID
            new_name: 新 Agent 名称
            new_description: 新 Agent 描述
            new_config: 新 Agent 配置（不传则复制源配置）
            created_by: 创建人 ID

        Returns:
            新 Agent 实例，源 Agent 不存在返回 None
        """
        try:
            # 获取源 Agent
            source_agent = await self.get(source_agent_id)
            if source_agent is None:
                logger.warning("source_agent_not_found", agent_id=source_agent_id)
                return None

            # 构建新 Agent 数据
            agent_data = {
                "id": str(uuid4()),
                "name": new_name,
                "description": new_description if new_description is not None else source_agent.description,
                "tenant_id": source_agent.tenant_id,
                "created_by": created_by,
                "config": new_config if new_config is not None else source_agent.config,
                "avatar": source_agent.avatar,
                "is_builtin": False,  # 复制的 Agent 不是内置的
            }

            # 创建新 Agent
            new_agent = Agent(**agent_data)
            self.session.add(new_agent)
            await self.session.flush()

            await self.session.commit()
            await self.session.refresh(new_agent)

            logger.info(
                "agent_copied",
                source_agent_id=source_agent_id,
                new_agent_id=new_agent.id,
                new_name=new_name,
            )
            return new_agent

        except Exception as e:
            await self.session.rollback()
            logger.error("agent_repository_copy_failed", source_agent_id=source_agent_id, error=str(e))
            raise

    async def list_by_tenant_with_tools(
        self,
        tenant_id: int | None = None,
        params: PaginationParams | None = None,
    ) -> PaginatedResult[Agent]:
        """按租户分页列出 Agent（包含工具关联）

        Args:
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        try:
            statement = select(Agent)

            if tenant_id is not None:
                statement = statement.where(Agent.tenant_id == tenant_id)

            # 排除已删除的
            statement = statement.where(Agent.deleted_at.is_(None))

            # 按创建时间倒序
            statement = statement.order_by(desc(Agent.created_at))

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
            logger.error("agent_repository_list_with_tools_failed", tenant_id=tenant_id, error=str(e))
            return PaginatedResult.create([], 0, params or PaginationParams())
