"""Agent 执行记录模型

用于 Multi-Agent 架构中追踪 Agent 调用链和执行状态。
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import Column as SAColumn
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import func
from sqlmodel import Column, Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.session import Session


class AgentExecutionBase(SQLModel):
    """Agent 执行基础模型"""

    session_id: str = Field(max_length=36)
    thread_id: str = Field(max_length=36)
    agent_id: str = Field(max_length=64)
    agent_type: str = Field(
        max_length=50,
        description="Agent 类型: supervisor, worker, router, leaf",
    )


class AgentExecution(AgentExecutionBase, table=True):
    """Agent 执行记录表模型

    用于追踪 Multi-Agent 调用链：
    - 记录每个 Agent 的执行过程
    - 支持 parent-child 关系（调用链）
    - 性能指标和调试支持
    """

    __tablename__ = "agent_executions"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)

    # 调用链关系
    parent_execution_id: UUID | None = Field(default=None)

    # 执行数据
    input_data: Any | None = Field(default=None, sa_column=Column(JSONB))
    output_data: Any | None = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(
        default="pending",
        max_length=20,
        description="执行状态: pending, running, completed, failed",
    )
    error_message: str | None = None

    # 性能指标
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None

    # 元数据（使用 meta_data 避免 SQLAlchemy 保留字冲突）
    meta_data: Any | None = Field(default=None, sa_column=Column(JSONB))

    # 时间戳（执行记录只需要 created_at，不需要 updated_at）
    created_at: datetime = Field(
        sa_column=SAColumn(
            DateTime(),
            server_default=func.now(),
            nullable=False,
        )
    )

    # 关系
    session: "Session" = Relationship(back_populates="agent_executions")
    children: list["AgentExecution"] = Relationship(
        back_populates="parent_execution"
    )
    parent_execution: "AgentExecution | None" = Relationship(
        back_populates="children"
    )


class AgentExecutionCreate(AgentExecutionBase):
    """Agent 执行创建模型"""

    parent_execution_id: UUID | None = None
    input_data: Any | None = None
    meta_data: Any | None = None  # 使用 meta_data 避免 SQLAlchemy 保留字冲突


class AgentExecutionUpdate(SQLModel):
    """Agent 执行更新模型"""

    output_data: Any | None = None
    status: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    meta_data: Any | None = None  # 使用 meta_data 避免 SQLAlchemy 保留字冲突


class AgentExecutionPublic(AgentExecutionBase):
    """Agent 执行公开信息"""

    id: UUID
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None




class AgentType:
    """Agent 类型常量"""

    SUPERVISOR = "supervisor"
    WORKER = "worker"
    ROUTER = "router"
    LEAF = "leaf"


class ExecutionStatus:
    """执行状态常量"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


__all__ = [
    "AgentExecution",
    "AgentExecutionBase",
    "AgentExecutionCreate",
    "AgentExecutionUpdate",
    "AgentExecutionPublic",
    "AgentType",
    "ExecutionStatus",
]
