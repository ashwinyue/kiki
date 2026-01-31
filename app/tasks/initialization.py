"""知识库初始化后台任务

使用 asyncio 后台任务执行知识库初始化。
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.infra.database import _get_session_factory
from app.observability.logging import get_logger
from app.schemas.knowledge_initialization import InitializationStatus
from app.services.knowledge.knowledge_initialization import (
    KnowledgeInitializationService,
)

logger = get_logger(__name__)


@dataclass
class InitializationTask:
    """初始化任务状态

    对齐 WeKnora99 的任务状态管理
    """

    task_id: str
    kb_id: str
    tenant_id: int
    status: InitializationStatus = InitializationStatus.PENDING
    message: str = "等待执行"
    progress_percent: float = 0.0
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "kb_id": self.kb_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "message": self.message,
            "progress_percent": self.progress_percent,
            "error": self.error,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


# 存储运行中的任务
_running_tasks: dict[str, asyncio.Task[Any]] = {}
# 存储任务状态
_task_status: dict[str, InitializationTask] = {}


async def start_initialization_task(
    task_id: str,
    kb_id: str,
    tenant_id: int,
    config: dict,
    session_factory: Any | None = None,
) -> None:
    """启动后台初始化任务

    Args:
        task_id: 任务 ID
        kb_id: 知识库 ID
        tenant_id: 租户 ID
        config: 初始化配置
        session_factory: 数据库会话工厂（可选）
    """
    if task_id in _running_tasks:
        logger.warning("initialization_task_already_running", task_id=task_id)
        return

    # 创建任务状态
    _task_status[task_id] = InitializationTask(
        task_id=task_id,
        kb_id=kb_id,
        tenant_id=tenant_id,
        status=InitializationStatus.PENDING,
        message="任务已创建",
        progress_percent=0.0,
    )

    async def run_initialization() -> None:
        try:
            # 使用全局会话工厂创建新会话
            factory = session_factory or _get_session_factory()
            async with factory() as session:
                # 更新状态为处理中
                _update_task_status(
                    task_id,
                    InitializationStatus.PROCESSING,
                    "开始初始化",
                    10.0,
                )

                # 导入配置 Schema（避免循环导入）
                from app.schemas.knowledge_initialization import (
                    InitializationConfig,
                )

                init_config = InitializationConfig(**config)

                # 执行初始化
                service = KnowledgeInitializationService(session)
                result = await service.initialize_kb(kb_id, tenant_id, init_config)

                # 更新最终状态
                if result.success:
                    _update_task_status(
                        task_id,
                        InitializationStatus.COMPLETED,
                        result.message,
                        100.0,
                    )
                else:
                    _update_task_status(
                        task_id,
                        InitializationStatus.FAILED,
                        result.message,
                        0.0,
                        result.error,
                    )

                logger.info(
                    "background_initialization_task_completed",
                    task_id=task_id,
                    kb_id=kb_id,
                    status=result.status.value,
                )

        except Exception as e:
            logger.error(
                "background_initialization_task_failed",
                task_id=task_id,
                kb_id=kb_id,
                error=str(e),
                exc_info=True,
            )
            _update_task_status(
                task_id,
                InitializationStatus.FAILED,
                f"初始化失败: {str(e)}",
                0.0,
                str(e),
            )
        finally:
            _running_tasks.pop(task_id, None)

    # 创建后台任务
    task = asyncio.create_task(run_initialization())
    _running_tasks[task_id] = task

    logger.info(
        "background_initialization_task_started",
        task_id=task_id,
        kb_id=kb_id,
        tenant_id=tenant_id,
    )


def is_task_running(task_id: str) -> bool:
    """检查任务是否正在运行

    Args:
        task_id: 任务 ID

    Returns:
        是否正在运行
    """
    task = _running_tasks.get(task_id)
    if task is None:
        return False
    return not task.done()


def cancel_task(task_id: str) -> bool:
    """取消初始化任务

    Args:
        task_id: 任务 ID

    Returns:
        是否成功取消
    """
    task = _running_tasks.get(task_id)
    if task is None:
        return False

    task.cancel()

    # 更新状态
    if task_id in _task_status:
        _task_status[task_id].status = InitializationStatus.CANCELLED
        _task_status[task_id].completed_at = datetime.now(UTC)

    _running_tasks.pop(task_id, None)

    logger.info("initialization_task_cancelled", task_id=task_id)
    return True


def get_running_tasks() -> list[str]:
    """获取正在运行的任务列表

    Returns:
        任务 ID 列表
    """
    return [
        task_id
        for task_id, task in _running_tasks.items()
        if not task.done()
    ]


def get_task_status(task_id: str) -> InitializationTask | None:
    """获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        任务状态，如果不存在则返回 None
    """
    return _task_status.get(task_id)


def _update_task_status(
    task_id: str,
    status: InitializationStatus,
    message: str,
    progress: float,
    error: str | None = None,
) -> None:
    """更新任务状态

    Args:
        task_id: 任务 ID
        status: 新状态
        message: 状态消息
        progress: 进度百分比
        error: 错误信息（可选）
    """
    if task_id not in _task_status:
        return

    task = _task_status[task_id]
    task.status = status
    task.message = message
    task.progress_percent = progress

    if error:
        task.error = error

    if status == InitializationStatus.PROCESSING and task.started_at is None:
        task.started_at = datetime.now(UTC)

    if status in (
        InitializationStatus.COMPLETED,
        InitializationStatus.FAILED,
        InitializationStatus.CANCELLED,
    ):
        task.completed_at = datetime.now(UTC)


__all__ = [
    "InitializationTask",
    "start_initialization_task",
    "is_task_running",
    "cancel_task",
    "get_running_tasks",
    "get_task_status",
]
