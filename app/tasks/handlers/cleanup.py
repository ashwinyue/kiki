"""清理任务处理器.

定期清理已完成的旧任务记录（Redis）。
"""

from app.observability.logging import get_logger
from app.tasks.store import TaskStore

logger = get_logger(__name__)


async def process_cleanup(days: int = 7) -> dict:
    """清理已完成旧任务

    Args:
        days: 保留天数

    Returns:
        清理结果
    """
    store = TaskStore()
    logger.info("cleanup_started", days=days)
    result = await store.cleanup(days=days)
    logger.info(
        "cleanup_completed",
        deleted_tasks=result.get("deleted_count", 0),
        deleted_logs=result.get("deleted_logs", 0),
    )
    return result
