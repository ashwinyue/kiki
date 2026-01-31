"""任务处理器基类和工具函数

提供任务状态管理和进度更新的通用功能。
"""

import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.task import Task, TaskCreate, TaskLog, TaskLogCreate
from app.observability.logging import get_logger

logger = get_logger(__name__)


class TaskHandler:
    """任务处理器基类

    提供任务状态管理和进度更新的通用功能。
    """

    def __init__(
        self,
        celery_task,
        session: AsyncSession,
        task_id: str,
        tenant_id: int,
    ):
        """初始化任务处理器

        Args:
            celery_task: Celery 任务实例
            session: 数据库会话
            task_id: 任务 ID
            tenant_id: 租户 ID
        """
        self.celery_task = celery_task
        self.session = session
        self.task_id = task_id
        self.tenant_id = tenant_id
        self._task_model: Task | None = None

    async def get_task(self) -> Task | None:
        """获取任务模型

        Returns:
            Task 模型实例
        """
        if self._task_model is None:
            stmt = select(Task).where(
                Task.task_id == self.task_id,
                Task.tenant_id == self.tenant_id,
            )
            result = await self.session.execute(stmt)
            self._task_model = result.scalar_one_or_none()
        return self._task_model

    async def create_task(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        title: str | None = None,
        description: str | None = None,
        business_id: str | None = None,
        business_type: str | None = None,
        total_items: int | None = None,
    ) -> Task:
        """创建任务记录

        Args:
            task_type: 任务类型
            payload: 任务参数
            title: 任务标题
            description: 任务描述
            business_id: 业务 ID
            business_type: 业务类型
            total_items: 总项目数

        Returns:
            创建的 Task 模型
        """
        task = Task(
            task_id=self.task_id,
            task_type=task_type,
            tenant_id=self.tenant_id,
            payload=payload,
            title=title,
            description=description,
            business_id=business_id,
            business_type=business_type,
            total_items=total_items,
            # 关联 Celery 任务 ID
            celery_task_id=self.celery_task.request.id,
        )

        self.session.add(task)
        await self.session.flush()

        self._task_model = task

        logger.info(
            "task_created",
            task_id=self.task_id,
            task_type=task_type,
            tenant_id=self.tenant_id,
        )

        return task

    async def update_progress(
        self,
        progress: int,
        current_step: str | None = None,
        processed_items: int | None = None,
        failed_items: int | None = None,
    ) -> None:
        """更新任务进度

        Args:
            progress: 进度百分比 (0-100)
            current_step: 当前步骤
            processed_items: 已处理项目数
            failed_items: 失败项目数
        """
        task = await self.get_task()
        if task:
            task.progress = progress
            if current_step:
                task.current_step = current_step
            if processed_items is not None:
                task.processed_items = processed_items
            if failed_items is not None:
                task.failed_items = failed_items
            task.updated_at = datetime.now(UTC)

    async def update_status(
        self,
        status: str,
        error_message: str | None = None,
        error_stack: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """更新任务状态

        Args:
            status: 新状态
            error_message: 错误信息
            error_stack: 错误堆栈
            result: 任务结果
        """
        task = await self.get_task()
        if task:
            task.status = status
            if error_message:
                task.error_message = error_message
            if error_stack:
                task.error_stack = error_stack
            if result:
                task.result = result
            task.updated_at = datetime.now(UTC)

            # 设置开始/完成时间
            if status == "processing" and not task.started_at:
                task.started_at = datetime.now(UTC)
            elif status in ("completed", "failed", "cancelled"):
                if not task.completed_at:
                    task.completed_at = datetime.now(UTC)

    async def add_log(
        self,
        level: str,
        message: str,
        step: str | None = None,
        item_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """添加任务日志

        Args:
            level: 日志级别
            message: 日志消息
            step: 步骤名称
            item_id: 关联的项目 ID
            extra_data: 额外数据
        """
        import uuid

        log = TaskLog(
            task_id=self.task_id,
            log_id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            level=level,
            message=message,
            step=step,
            item_id=item_id,
            extra_data=extra_data,
        )

        self.session.add(log)

    async def mark_started(self) -> None:
        """标记任务开始"""
        await self.update_status("processing")

    async def mark_completed(self, result: dict[str, Any] | None = None) -> None:
        """标记任务完成

        Args:
            result: 任务结果
        """
        await self.update_progress(100)
        await self.update_status("completed", result=result)

    async def mark_failed(
        self,
        error_message: str,
        error_stack: str | None = None,
    ) -> None:
        """标记任务失败

        Args:
            error_message: 错误信息
            error_stack: 错误堆栈
        """
        await self.update_status("failed", error_message=error_message, error_stack=error_stack)

    async def increment_retry(self) -> int:
        """增加重试计数

        Returns:
            新的重试次数
        """
        task = await self.get_task()
        if task:
            task.retry_count += 1
            task.updated_at = datetime.now(UTC)
            return task.retry_count
        return 0


# ============== 任务上下文管理器 ==============


class task_context:
    """任务上下文管理器

    自动管理任务状态和错误处理。

    Examples:
        async with task_context(celery_task, session, task_id, tenant_id) as handler:
            await handler.mark_started()

            for i in range(10):
                await handler.update_progress(i * 10, f"Processing {i}")
                # ... 处理逻辑

            await handler.mark_completed({"count": 10})
    """

    def __init__(
        self,
        celery_task,
        session: AsyncSession,
        task_id: str,
        tenant_id: int,
    ):
        self.celery_task = celery_task
        self.session = session
        self.task_id = task_id
        self.tenant_id = tenant_id
        self._handler: TaskHandler | None = None

    async def __aenter__(self) -> TaskHandler:
        self._handler = TaskHandler(
            self.celery_task,
            self.session,
            self.task_id,
            self.tenant_id,
        )
        return self._handler

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 任务执行失败
            import traceback

            error_message = str(exc_val)
            error_stack = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))

            if self._handler:
                await self._handler.mark_failed(error_message, error_stack)

            logger.error(
                "task_failed",
                task_id=self.task_id,
                error=error_message,
            )


# ============== 进度报告工具 ==============


class ProgressReporter:
    """进度报告器

    提供便捷的进度更新接口。

    Examples:
        reporter = ProgressReporter(handler, total=100)

        for i in range(100):
            # ... 处理逻辑
            await reporter.increment(item_id=str(i))
    """

    def __init__(
        self,
        handler: TaskHandler,
        total: int,
        current_step: str | None = None,
    ):
        self.handler = handler
        self.total = total
        self.current_step = current_step
        self.processed = 0
        self.failed = 0

    async def increment(
        self,
        item_id: str | None = None,
        success: bool = True,
    ) -> None:
        """增加进度

        Args:
            item_id: 项目 ID
            success: 是否成功
        """
        self.processed += 1
        if not success:
            self.failed += 1

        progress = int((self.processed / self.total) * 100)
        await self.handler.update_progress(
            progress=progress,
            current_step=self.current_step,
            processed_items=self.processed,
            failed_items=self.failed,
        )

    async def set_step(self, step: str) -> None:
        """设置当前步骤

        Args:
            step: 步骤名称
        """
        self.current_step = step
        await self.handler.update_progress(
            progress=int((self.processed / self.total) * 100),
            current_step=step,
        )


# ============== 导出 ==============

__all__ = [
    "TaskHandler",
    "task_context",
    "ProgressReporter",
]
