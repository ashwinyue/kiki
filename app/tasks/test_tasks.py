"""测试任务模块

提供异步模型测试任务，支持进度跟踪。

对齐 WeKnora99 的异步测试任务系统。
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.services.model_test import ModelTestService, TestStatus
from app.tasks.types import TaskStatus

logger = get_logger(__name__)


# ============== 测试任务类型 ==============


class TestTaskType(str, Enum):
    """测试任务类型"""

    EMBEDDING = "embedding"
    RERANK = "rerank"
    LLM = "llm"
    MULTIMODAL = "multimodal"


# ============== 测试任务数据类 ==============


@dataclass
class TestTask:
    """测试任务"""

    task_id: str
    task_type: TestTaskType
    tenant_id: int
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.task_id,
            "type": self.task_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "createdAt": self.created_at.isoformat(),
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }


# ============== 测试任务管理器 ==============


class TestTaskManager:
    """测试任务管理器

    管理异步测试任务的创建、执行、进度跟踪。
    """

    def __init__(self) -> None:
        """初始化任务管理器"""
        self._tasks: dict[str, TestTask] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

    def create_task(
        self,
        task_type: TestTaskType,
        tenant_id: int,
        parameters: dict[str, Any],
    ) -> TestTask:
        """创建测试任务

        Args:
            task_type: 任务类型
            tenant_id: 租户 ID
            parameters: 任务参数

        Returns:
            创建的任务
        """
        task_id = str(uuid.uuid4())

        task = TestTask(
            task_id=task_id,
            task_type=task_type,
            tenant_id=tenant_id,
            parameters=parameters,
            message="任务已创建，等待执行",
        )

        self._tasks[task_id] = task

        logger.info(
            "test_task_created",
            task_id=task_id,
            task_type=task_type.value,
            tenant_id=tenant_id,
        )

        return task

    async def execute_task(
        self,
        task_id: str,
        session: AsyncSession,
    ) -> None:
        """执行测试任务

        Args:
            task_id: 任务 ID
            session: 数据库会话
        """
        task = self._tasks.get(task_id)

        if not task:
            logger.warning("test_task_not_found", task_id=task_id)
            return

        if task.status != TaskStatus.PENDING:
            logger.warning(
                "test_task_not_pending",
                task_id=task_id,
                status=task.status.value,
            )
            return

        # 更新任务状态
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now(UTC)
        task.progress = 0.0
        task.message = "正在测试..."

        logger.info(
            "test_task_started",
            task_id=task_id,
            task_type=task.task_type.value,
        )

        try:
            test_service = ModelTestService(session)

            # 根据任务类型执行测试
            if task.task_type == TestTaskType.EMBEDDING:
                await self._execute_embedding_test(task, test_service)
            elif task.task_type == TestTaskType.RERANK:
                await self._execute_rerank_test(task, test_service)
            elif task.task_type == TestTaskType.LLM:
                await self._execute_llm_test(task, test_service)
            elif task.task_type == TestTaskType.MULTIMODAL:
                await self._execute_multimodal_test(task, test_service)
            else:
                raise ValueError(f"不支持的测试类型: {task.task_type}")

            # 任务完成
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            task.completed_at = datetime.now(UTC)

            logger.info(
                "test_task_completed",
                task_id=task_id,
                status=task.status.value,
            )

        except Exception as e:
            # 任务失败
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.message = f"测试失败: {str(e)}"
            task.completed_at = datetime.now(UTC)

            logger.error(
                "test_task_failed",
                task_id=task_id,
                error=str(e),
            )

    async def _execute_embedding_test(
        self,
        task: TestTask,
        test_service: ModelTestService,
    ) -> None:
        """执行 Embedding 测试

        Args:
            task: 测试任务
            test_service: 测试服务
        """
        params = task.parameters

        task.progress = 20.0
        task.message = "正在连接 Embedding 模型..."

        result = await test_service.test_embedding(
            source=params.get("source", "remote"),
            model_name=params.get("model_name", ""),
            base_url=params.get("base_url", ""),
            api_key=params.get("api_key", ""),
            dimension=params.get("dimension", 0),
            provider=params.get("provider", ""),
        )

        task.progress = 100.0

        if result.available:
            task.message = f"测试成功，向量维度: {result.dimension}"
            task.result = {
                "available": True,
                "dimension": result.dimension,
                "message": result.message,
            }
        else:
            task.message = result.message
            task.result = {
                "available": False,
                "dimension": 0,
                "message": result.message,
            }

    async def _execute_rerank_test(
        self,
        task: TestTask,
        test_service: ModelTestService,
    ) -> None:
        """执行 Rerank 测试

        Args:
            task: 测试任务
            test_service: 测试服务
        """
        params = task.parameters

        task.progress = 20.0
        task.message = "正在连接 Rerank 模型..."

        result = await test_service.test_rerank(
            model_name=params.get("model_name", ""),
            base_url=params.get("base_url", ""),
            api_key=params.get("api_key", ""),
            provider=params.get("provider", ""),
        )

        task.progress = 100.0

        if result.available:
            task.message = "Rerank 模型测试成功"
            task.result = {
                "available": True,
                "message": result.message,
            }
        else:
            task.message = result.message
            task.result = {
                "available": False,
                "message": result.message,
            }

    async def _execute_llm_test(
        self,
        task: TestTask,
        test_service: ModelTestService,
    ) -> None:
        """执行 LLM 测试

        Args:
            task: 测试任务
            test_service: 测试服务
        """
        params = task.parameters

        task.progress = 20.0
        task.message = "正在连接 LLM 模型..."

        result = await test_service.test_llm(
            model_name=params.get("model_name", ""),
            base_url=params.get("base_url", ""),
            api_key=params.get("api_key", ""),
            provider=params.get("provider", ""),
        )

        task.progress = 100.0

        if result.status == TestStatus.SUCCESS:
            task.message = f"LLM 测试成功，延迟: {result.latency_ms}ms"
            task.result = {
                "available": True,
                "latency_ms": result.latency_ms,
                "message": result.message,
                "details": result.details,
            }
        else:
            task.message = result.message
            task.result = {
                "available": False,
                "message": result.message,
                "status": result.status.value,
            }

    async def _execute_multimodal_test(
        self,
        task: TestTask,
        test_service: ModelTestService,
    ) -> None:
        """执行多模态测试

        Args:
            task: 测试任务
            test_service: 测试服务
        """
        params = task.parameters

        task.progress = 20.0
        task.message = "正在连接多模态模型..."

        result = await test_service.test_multimodal(
            model_name=params.get("model_name", ""),
            base_url=params.get("base_url", ""),
            api_key=params.get("api_key", ""),
            image_base64=params.get("image_base64"),
        )

        task.progress = 100.0

        if result.status == TestStatus.SUCCESS:
            task.message = f"多模态测试成功，延迟: {result.latency_ms}ms"
            task.result = {
                "available": True,
                "latency_ms": result.latency_ms,
                "message": result.message,
                "details": result.details,
            }
        else:
            task.message = result.message
            task.result = {
                "available": False,
                "message": result.message,
                "status": result.status.value,
            }

    def start_task(
        self,
        task_id: str,
        session: AsyncSession,
    ) -> bool:
        """启动测试任务（后台执行）

        Args:
            task_id: 任务 ID
            session: 数据库会话

        Returns:
            是否成功启动
        """
        task = self._tasks.get(task_id)

        if not task:
            logger.warning("test_task_not_found", task_id=task_id)
            return False

        if task_id in self._running_tasks:
            logger.warning("test_task_already_running", task_id=task_id)
            return False

        # 创建后台任务
        async_task = asyncio.create_task(
            self.execute_task(task_id, session),
            name=f"test_task_{task_id}",
        )

        self._running_tasks[task_id] = async_task

        # 添加完成回调
        async_task.add_done_callback(
            lambda t: self._on_task_done(task_id, t)
        )

        logger.info("test_task_started_background", task_id=task_id)

        return True

    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """任务完成回调

        Args:
            task_id: 任务 ID
            task: 异步任务
        """
        self._running_tasks.pop(task_id, None)

        if task.exception():
            logger.error(
                "test_task_exception",
                task_id=task_id,
                error=str(task.exception()),
            )

    def get_task(self, task_id: str) -> TestTask | None:
        """获取测试任务

        Args:
            task_id: 任务 ID

        Returns:
            任务对象
        """
        return self._tasks.get(task_id)

    def get_tasks_by_tenant(self, tenant_id: int) -> list[TestTask]:
        """获取租户的所有测试任务

        Args:
            tenant_id: 租户 ID

        Returns:
            任务列表
        """
        return [
            task for task in self._tasks.values()
            if task.tenant_id == tenant_id
        ]

    def cancel_task(self, task_id: str) -> bool:
        """取消测试任务

        Args:
            task_id: 任务 ID

        Returns:
            是否成功取消
        """
        task = self._tasks.get(task_id)

        if not task:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        # 取消运行中的任务
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            self._running_tasks.pop(task_id, None)

        task.status = TaskStatus.CANCELLED
        task.message = "任务已取消"
        task.completed_at = datetime.now(UTC)

        logger.info("test_task_cancelled", task_id=task_id)

        return True

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """清理旧任务

        Args:
            max_age_hours: 最大保留时间（小时）

        Returns:
            清理的任务数量
        """
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)

        to_remove = [
            task_id for task_id, task in self._tasks.items()
            if task.created_at < cutoff
            and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]

        for task_id in to_remove:
            self._tasks.pop(task_id, None)

        if to_remove:
            logger.info(
                "test_tasks_cleaned",
                count=len(to_remove),
                max_age_hours=max_age_hours,
            )

        return len(to_remove)


# ============== 全局任务管理器 ==============

_test_task_manager: TestTaskManager | None = None


def get_test_task_manager() -> TestTaskManager:
    """获取全局测试任务管理器（单例）

    Returns:
        TestTaskManager 实例
    """
    global _test_task_manager
    if _test_task_manager is None:
        _test_task_manager = TestTaskManager()
    return _test_task_manager


__all__ = [
    # 枚举
    "TestTaskType",
    # 数据类
    "TestTask",
    # 管理器
    "TestTaskManager",
    "get_test_task_manager",
]
