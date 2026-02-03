"""Agent Execution Repository

提供 Agent 执行记录的数据访问层。
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlmodel import Session

from app.models.agent_execution import (
    AgentExecution,
    AgentExecutionCreate,
    AgentType,
    ExecutionStatus,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AgentExecutionRepository:
    """Agent 执行记录仓储

    提供 CRUD 操作和调用链查询。
    """

    def __init__(self, session: Session):
        """初始化仓储

        Args:
            session: 数据库会话
        """
        self.session = session

    async def create(
        self,
        execution: AgentExecutionCreate,
    ) -> AgentExecution:
        """创建执行记录

        Args:
            execution: 执行创建数据

        Returns:
            创建的执行记录
        """
        db_execution = AgentExecution.from_orm(execution)

        self.session.add(db_execution)
        await self.session.flush()
        await self.session.refresh(db_execution)

        logger.debug(
            "agent_execution_created",
            execution_id=str(db_execution.id),
            agent_id=db_execution.agent_id,
            agent_type=db_execution.agent_type,
        )

        return db_execution

    async def get_by_id(self, execution_id: UUID) -> AgentExecution | None:
        """根据 ID 获取执行记录

        Args:
            execution_id: 执行记录 ID

        Returns:
            执行记录或 None
        """
        statement = select(AgentExecution).where(
            AgentExecution.id == execution_id
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def update(
        self,
        execution: AgentExecution,
    ) -> AgentExecution:
        """更新执行记录

        Args:
            execution: 要更新的执行记录

        Returns:
            更新后的执行记录
        """
        self.session.add(execution)
        await self.session.flush()
        await self.session.refresh(execution)

        return execution

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[AgentExecution]:
        """列出会话的所有执行记录

        Args:
            session_id: 会话 ID
            limit: 返回数量限制

        Returns:
            执行记录列表（按创建时间倒序）
        """
        statement = (
            select(AgentExecution)
            .where(AgentExecution.session_id == session_id)
            .order_by(AgentExecution.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_execution_chain(
        self,
        execution_id: UUID,
        max_depth: int = 10,
    ) -> list[AgentExecution]:
        """获取执行调用链（递归查询）

        Args:
            execution_id: 顶层执行记录 ID
            max_depth: 最大递归深度

        Returns:
            调用链列表（从顶层到底层）
        """
        chain = []
        visited = set()

        async def _fetch_parent(current_id: UUID, depth: int):
            if depth > max_depth:
                return
            if current_id in visited:
                return  # 防止循环

            visited.add(current_id)
            execution = await self.get_by_id(current_id)
            if execution:
                chain.append(execution)

                # 递归获取父执行
                if execution.parent_execution_id:
                    await _fetch_parent(execution.parent_execution_id, depth + 1)

        await _fetch_parent(execution_id, 0)

        return chain

    async def list_children(
        self,
        parent_execution_id: UUID,
    ) -> list[AgentExecution]:
        """列出子执行记录

        Args:
            parent_execution_id: 父执行记录 ID

        Returns:
            子执行记录列表
        """
        statement = select(AgentExecution).where(
            AgentExecution.parent_execution_id == parent_execution_id
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_agent(
        self,
        agent_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentExecution]:
        """列出指定 Agent 的执行记录

        Args:
            agent_id: Agent ID
            session_id: 可选的会话 ID 过滤
            limit: 返回数量限制

        Returns:
            执行记录列表
        """
        statement = select(AgentExecution).where(
            AgentExecution.agent_id == agent_id
        )

        if session_id:
            statement = statement.where(
                AgentExecution.session_id == session_id
            )

        statement = statement.order_by(AgentExecution.created_at.desc()).limit(
            limit
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_execution_stats(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """获取会话的执行统计信息

        Args:
            session_id: 会话 ID

        Returns:
            统计信息字典
        """
        executions = await self.list_by_session(session_id, limit=1000)

        total = len(executions)
        completed = sum(1 for e in executions if e.status == "completed")
        failed = sum(1 for e in executions if e.status == "failed")
        running = sum(1 for e in executions if e.status == "running")

        # 计算 avg duration
        completed_with_duration = [
            e.duration_ms
            for e in executions
            if e.duration_ms is not None
        ]
        avg_duration = (
            sum(completed_with_duration) / len(completed_with_duration)
            if completed_with_duration
            else 0
        )

        # Agent 调用统计
        agent_counts: dict[str, int] = {}
        for e in executions:
            agent_counts[e.agent_id] = agent_counts.get(e.agent_id, 0) + 1

        return {
            "total_executions": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "avg_duration_ms": avg_duration,
            "agent_counts": agent_counts,
        }


class AgentExecutionTracker:
    """Agent 调用链追踪服务

    在 Multi-Agent 执行过程中自动记录调用链。
    """

    def __init__(self, session: Session):
        """初始化追踪器

        Args:
            session: 数据库会话
        """
        self.session = session
        self.repository = AgentExecutionRepository(session)
        self._current_execution_id: UUID | None = None

    async def start_execution(
        self,
        session_id: str,
        thread_id: str,
        agent_id: str,
        agent_type: str,
        input_data: dict[str, Any],
        parent_execution_id: UUID | None = None,
        meta_data: dict[str, Any] | None = None,  # 重命名避免 SQLAlchemy 保留字冲突
    ) -> AgentExecution:
        """开始 Agent 执行并记录

        Args:
            session_id: 会话 ID
            thread_id: 线程 ID
            agent_id: Agent ID
            agent_type: Agent 类型
            input_data: 输入数据
            parent_execution_id: 父执行 ID（用于调用链）
            meta_data: 元数据

        Returns:
            创建的执行记录
        """
        execution_create = AgentExecutionCreate(
            session_id=session_id,
            thread_id=thread_id,
            agent_id=agent_id,
            agent_type=agent_type,
            parent_execution_id=parent_execution_id,
            input_data=input_data,
            meta_data=meta_data,
            status="pending",
            started_at=datetime.now(UTC),
        )

        execution = await self.repository.create(execution_create)
        self._current_execution_id = execution.id

        logger.info(
            "agent_execution_started",
            execution_id=str(execution.id),
            agent_id=agent_id,
            parent_execution_id=str(parent_execution_id) if parent_execution_id else None,
        )

        return execution

    async def complete_execution(
        self,
        execution_id: UUID,
        output_data: dict[str, Any],
        error_message: str | None = None,
    ) -> AgentExecution:
        """完成 Agent 执行

        Args:
            execution_id: 执行记录 ID
            output_data: 输出数据
            error_message: 错误信息（如果有）

        Returns:
            更新后的执行记录
        """
        execution = await self.repository.get_by_id(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        # 计算执行时长
        if execution.started_at:
            completed_at = datetime.now(UTC)
            duration_ms = int(
                (completed_at - execution.started_at).total_seconds() * 1000
            )
        else:
            completed_at = datetime.now(UTC)
            duration_ms = None

        # 更新状态
        execution.output_data = output_data
        execution.completed_at = completed_at
        execution.duration_ms = duration_ms
        execution.status = (
            "completed" if error_message is None else "failed"
        )
        execution.error_message = error_message

        execution = await self.repository.update(execution)

        logger.info(
            "agent_execution_completed",
            execution_id=str(execution.id),
            agent_id=execution.agent_id,
            status=execution.status,
            duration_ms=duration_ms,
        )

        return execution

    async def complete_current_execution(
        self,
        output_data: dict[str, Any],
        error_message: str | None = None,
    ) -> AgentExecution | None:
        """完成当前执行的 Agent

        Args:
            output_data: 输出数据
            error_message: 错误信息

        Returns:
            更新后的执行记录，如果没有当前执行则返回 None
        """
        if self._current_execution_id is None:
            return None

        return await self.complete_execution(
            self._current_execution_id,
            output_data,
            error_message,
        )

    @property
    def current_execution_id(self) -> UUID | None:
        """获取当前执行 ID"""
        return self._current_execution_id


__all__ = [
    # Repository
    "AgentExecutionRepository",
    # 追踪服务
    "AgentExecutionTracker",
]
