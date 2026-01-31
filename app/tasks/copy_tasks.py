"""知识库复制后台任务

使用 asyncio 后台任务执行知识库复制。
"""

import asyncio
from typing import Any

from app.infra.database import _get_session_factory
from app.observability.logging import get_logger
from app.services.knowledge.knowledge_clone import execute_copy_task

logger = get_logger(__name__)

# 存储运行中的任务
_running_tasks: dict[str, asyncio.Task[Any]] = {}


async def start_copy_task(
    task_id: str,
    session_factory: Any | None = None,
) -> None:
    """启动后台复制任务

    Args:
        task_id: 任务 ID
        session_factory: 数据库会话工厂（可选）
    """
    if task_id in _running_tasks:
        logger.warning("copy_task_already_running", task_id=task_id)
        return

    async def run_copy() -> None:
        try:
            # 使用全局会话工厂创建新会话
            factory = session_factory or _get_session_factory()
            async with factory() as session:
                result = await execute_copy_task(session, task_id)
                logger.info(
                    "background_copy_task_completed",
                    task_id=task_id,
                    status=result.status,
                    progress_percent=result.to_dict()["progress_percent"],
                )
        except Exception as e:
            logger.error(
                "background_copy_task_failed",
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )
        finally:
            _running_tasks.pop(task_id, None)

    # 创建后台任务
    task = asyncio.create_task(run_copy())
    _running_tasks[task_id] = task

    logger.info("background_copy_task_started", task_id=task_id)


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
    """取消复制任务

    Args:
        task_id: 任务 ID

    Returns:
        是否成功取消
    """
    task = _running_tasks.get(task_id)
    if task is None:
        return False

    task.cancel()
    _running_tasks.pop(task_id, None)

    logger.info("copy_task_cancelled", task_id=task_id)
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


__all__ = [
    "start_copy_task",
    "is_task_running",
    "cancel_task",
    "get_running_tasks",
]
