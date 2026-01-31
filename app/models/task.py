"""异步任务状态模型.

对齐 WeKnora99 的任务状态管理，支持任务状态与进度追踪。
任务状态当前由 Redis 持久化，SQLModel 主要用于 API Schema 复用。
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.tasks.types import TaskPriority, TaskStatus


class TaskBase(SQLModel):
    """任务基础模型"""

    task_id: str = Field(max_length=200, description="任务 ID", primary_key=True)
    task_type: str = Field(max_length=100, description="任务类型")
    tenant_id: int = Field(description="租户 ID")
    priority: TaskPriority = Field(
        default=TaskPriority.DEFAULT,
        description="任务优先级",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="任务状态",
    )

    # 任务描述
    title: str | None = Field(None, max_length=500, description="任务标题")
    description: str | None = Field(None, max_length=2000, description="任务描述")

    # 任务参数 (JSON 格式)
    payload: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="任务参数",
    )

    # 进度信息
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比 (0-100)")
    current_step: str | None = Field(None, max_length=200, description="当前步骤")

    # 统计信息
    total_items: int | None = Field(default=None, description="总项目数")
    processed_items: int | None = Field(default=None, description="已处理项目数")
    failed_items: int | None = Field(default=None, description="失败项目数")

    # 结果信息
    result: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="任务结果",
    )

    # 错误信息
    error_message: str | None = Field(None, max_length=5000, description="错误信息")
    error_stack: str | None = Field(None, description="错误堆栈")

    # Celery 任务 ID (用于关联 Celery 任务)
    celery_task_id: str | None = Field(None, max_length=255, description="Celery 任务 ID")

    # 重试信息
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")


class Task(TaskBase, table=True):
    """任务表模型

    存储所有异步任务的状态和进度信息。
    """

    __tablename__ = "tasks"

    # 业务相关 ID (可选，用于关联具体业务对象)
    business_id: str | None = Field(
        default=None,
        max_length=100,
        description="业务 ID (如知识库 ID)",
    )
    business_type: str | None = Field(
        default=None,
        max_length=50,
        description="业务类型 (如 knowledge_base, knowledge)",
    )

    # 父任务 ID (用于任务链)
    parent_task_id: str | None = Field(
        default=None,
        max_length=200,
        description="父任务 ID",
    )

    # 额外元数据
    extra_metadata: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="额外元数据",
    )

    # 时间戳
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = Field(default=None, description="开始执行时间")
    completed_at: datetime | None = Field(default=None, description="完成时间")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 创建者信息
    created_by: str | None = Field(default=None, max_length=36, description="创建者 ID")

    # 软删除
    deleted_at: datetime | None = Field(default=None)

    @property
    def duration(self) -> float | None:
        """任务执行时长 (秒)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        if self.started_at:
            return (datetime.now(UTC) - self.started_at).total_seconds()
        return None

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self.status == TaskStatus.PROCESSING

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def is_failed(self) -> bool:
        """是否失败"""
        return self.status == TaskStatus.FAILED

    @property
    def can_retry(self) -> bool:
        """是否可以重试"""
        return (
            self.status == TaskStatus.FAILED
            and self.retry_count < self.max_retries
        )


class TaskCreate(SQLModel):
    """任务创建模型"""

    task_id: str
    task_type: str
    tenant_id: int
    priority: TaskPriority = TaskPriority.DEFAULT
    title: str | None = None
    description: str | None = None
    payload: dict[str, Any] | None = None
    business_id: str | None = None
    business_type: str | None = None
    parent_task_id: str | None = None
    max_retries: int = 3
    total_items: int | None = None
    created_by: str | None = None


class TaskUpdate(SQLModel):
    """任务更新模型"""

    status: TaskStatus | None = None
    progress: int | None = None
    current_step: str | None = None
    processed_items: int | None = None
    failed_items: int | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None
    error_stack: str | None = None
    celery_task_id: str | None = None
    retry_count: int | None = None
    extra_metadata: dict[str, Any] | None = None


class TaskPublic(SQLModel):
    """任务公开信息模型"""

    task_id: str
    task_type: str
    tenant_id: int
    priority: TaskPriority
    status: TaskStatus
    title: str | None
    description: str | None
    progress: int
    current_step: str | None
    total_items: int | None
    processed_items: int | None
    failed_items: int | None
    result: dict[str, Any] | None
    error_message: str | None
    retry_count: int
    max_retries: int
    business_id: str | None
    business_type: str | None
    parent_task_id: str | None
    duration: float | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class TaskList(SQLModel):
    """任务列表响应模型"""

    items: list[TaskPublic]
    total: int
    page: int
    size: int


# ============== 任务日志模型 ==============


class TaskLogBase(SQLModel):
    """任务日志基础模型"""

    task_id: str = Field(
        max_length=200,
        description="任务 ID",
        primary_key=True,
    )
    log_id: str = Field(max_length=100, description="日志 ID")

    level: str = Field(max_length=20, description="日志级别: DEBUG, INFO, WARNING, ERROR")
    message: str = Field(description="日志消息")

    # 关联信息
    step: str | None = Field(None, max_length=200, description="步骤名称")
    item_id: str | None = Field(None, max_length=100, description="关联的项目 ID")

    # 额外数据
    extra_data: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="额外数据",
    )


class TaskLog(TaskLogBase, table=True):
    """任务日志表模型"""

    __tablename__ = "task_logs"

    tenant_id: int = Field(description="租户 ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskLogCreate(SQLModel):
    """任务日志创建模型"""

    task_id: str
    level: str = "INFO"
    message: str
    step: str | None = None
    item_id: str | None = None
    extra_data: dict[str, Any] | None = None


class TaskLogPublic(SQLModel):
    """任务日志公开信息模型"""

    task_id: str
    log_id: str
    level: str
    message: str
    step: str | None
    item_id: str | None
    extra_data: dict[str, Any] | None
    created_at: datetime


# ============== 导出 ==============

__all__ = [
    # 任务模型
    "Task",
    "TaskBase",
    "TaskCreate",
    "TaskUpdate",
    "TaskPublic",
    "TaskList",
    # 任务日志模型
    "TaskLog",
    "TaskLogBase",
    "TaskLogCreate",
    "TaskLogPublic",
]
