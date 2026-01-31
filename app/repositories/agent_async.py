"""Agent 异步仓储模块

提供 Agent 的异步数据访问层，兼容 BaseRepository 模式。
"""

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent, AgentCreate, AgentExecution, AgentStatus, AgentType, AgentUpdate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


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

    async def create_with_tools(
        self,
        data: AgentCreate,
    ) -> Agent:
        """创建 Agent

        Args:
            data: Agent 创建数据

        Returns:
            创建的 Agent
        """
        agent = Agent(**data.model_dump())
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

    async def list_by_type_and_status(
        self,
        agent_type: AgentType | None = None,
        status: AgentStatus | None = None,
        params: PaginationParams | None = None,
        tenant_id: int | None = None,
    ) -> PaginatedResult[Agent]:
        """根据类型和状态分页列出 Agent

        Args:
            agent_type: 筛选类型
            status: 筛选状态
            params: 分页参数

        Returns:
            分页结果
        """
        try:
            statement = select(Agent)

            if tenant_id is not None:
                statement = statement.where(Agent.tenant_id == tenant_id)
            if agent_type:
                statement = statement.where(Agent.agent_type == agent_type)
            if status:
                statement = statement.where(Agent.status == status)

            # 排除已删除的
            statement = statement.where(Agent.status != AgentStatus.DELETED)

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
            logger.error(
                "agent_repository_list_failed",
                agent_type=agent_type,
                status=status,
                error=str(e),
            )
            return PaginatedResult.create([], 0, params or PaginationParams())

    async def update_agent(
        self,
        agent_id: int,
        data: AgentUpdate,
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
            update_data = data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
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

    async def soft_delete(self, agent_id: int) -> bool:
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

            agent.status = AgentStatus.DELETED
            await self.session.commit()

            logger.info("agent_soft_deleted", agent_id=agent_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("agent_repository_delete_failed", agent_id=agent_id, error=str(e))
            return False

    async def get_active_count(self) -> int:
        """获取活跃 Agent 数量

        Returns:
            活跃 Agent 数量
        """
        try:
            from sqlalchemy import func

            statement = select(func.count()).select_from(Agent).where(
                Agent.status == AgentStatus.ACTIVE
            )
            result = await self.session.execute(statement)
            return result.scalar() or 0

        except Exception as e:
            logger.error("agent_repository_count_failed", error=str(e))
            return 0

    async def get_active_count_by_tenant(self, tenant_id: int) -> int:
        """获取指定租户的活跃 Agent 数量"""
        try:
            from sqlalchemy import func

            statement = (
                select(func.count())
                .select_from(Agent)
                .where(Agent.status == AgentStatus.ACTIVE)
                .where(Agent.tenant_id == tenant_id)
            )
            result = await self.session.execute(statement)
            return result.scalar() or 0
        except Exception as e:
            logger.error("agent_repository_count_failed", tenant_id=tenant_id, error=str(e))
            return 0

    async def list_ids_by_tenant(self, tenant_id: int) -> list[int]:
        """列出租户下所有 Agent ID"""
        try:
            statement = select(Agent.id).where(Agent.tenant_id == tenant_id)
            result = await self.session.execute(statement)
            return [row[0] for row in result.all() if row[0] is not None]
        except Exception as e:
            logger.error("agent_repository_list_ids_failed", tenant_id=tenant_id, error=str(e))
            return []


class AgentExecutionRepositoryAsync:
    """Agent 执行历史异步仓储"""

    def __init__(self, session: AsyncSession) -> None:
        """初始化仓储

        Args:
            session: 异步数据库会话
        """
        self.session = session

    async def list_by_agent(
        self,
        agent_id: int,
        limit: int = 50,
    ) -> list[AgentExecution]:
        """列出 Agent 的执行记录

        Args:
            agent_id: Agent ID
            limit: 限制数量

        Returns:
            执行记录列表
        """
        try:
            statement = (
                select(AgentExecution)
                .where(AgentExecution.agent_id == agent_id)
                .order_by(desc(AgentExecution.created_at))
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "agent_execution_repository_list_by_agent_failed",
                agent_id=agent_id,
                error=str(e),
            )
            return []

    async def list_by_agents(
        self,
        agent_ids: list[int],
        limit: int = 50,
    ) -> list[AgentExecution]:
        """列出多个 Agent 的执行记录"""
        if not agent_ids:
            return []
        try:
            statement = (
                select(AgentExecution)
                .where(AgentExecution.agent_id.in_(agent_ids))
                .order_by(AgentExecution.created_at.desc())
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("agent_execution_list_by_agents_failed", error=str(e))
            return []

    async def list_recent(
        self,
        limit: int = 20,
    ) -> list[AgentExecution]:
        """列出最近的执行记录

        Args:
            limit: 限制数量

        Returns:
            执行记录列表
        """
        try:
            statement = (
                select(AgentExecution)
                .order_by(desc(AgentExecution.created_at))
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return list(result.scalars().all())

        except Exception as e:
            logger.error("agent_execution_repository_list_recent_failed", error=str(e))
            return []
