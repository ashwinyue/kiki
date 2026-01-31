"""清理任务处理器

定期清理已完成的旧任务记录。
"""

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.task import Task, TaskLog
from app.observability.logging import get_logger

logger = get_logger(__name__)


async def process_cleanup(days: int = 7) -> dict:
    """清理已完成旧任务

    Args:
        days: 保留天数

    Returns:
        清理结果
    """
    from app.infra.database import get_session_factory

    session_factory = get_session_factory()

    async with session_factory() as session:
        cutoff_time = datetime.now(UTC) - timedelta(days=days)

        logger.info(
            "cleanup_started",
            days=days,
            cutoff_time=cutoff_time.isoformat(),
        )

        # 统计需要删除的任务
        stmt = select(Task).where(
            Task.completed_at < cutoff_time,
            Task.status.in_(["completed", "failed", "cancelled"]),
        )
        result = await session.execute(stmt)
        tasks_to_delete = result.scalars().all()

        task_count = len(tasks_to_delete)
        task_ids = [t.task_id for t in tasks_to_delete]

        if task_count == 0:
            logger.info("cleanup_no_tasks_found")
            return {"deleted_count": 0}

        # 删除任务日志
        log_stmt = delete(TaskLog).where(TaskLog.task_id.in_(task_ids))
        log_result = await session.execute(log_stmt)
        log_count = log_result.rowcount

        # 删除任务记录
        task_stmt = delete(Task).where(Task.task_id.in_(task_ids))
        task_result = await session.execute(task_stmt)

        await session.commit()

        logger.info(
            "cleanup_completed",
            deleted_tasks=task_count,
            deleted_logs=log_count,
        )

        return {
            "deleted_count": task_count,
            "deleted_logs": log_count,
            "cutoff_time": cutoff_time.isoformat(),
        }
