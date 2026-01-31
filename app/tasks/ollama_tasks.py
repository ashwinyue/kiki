"""Ollama 模型下载任务

对齐 WeKnora99 的异步模型下载任务实现
(internal/handler/initialization.go - DownloadOllamaModel/downloadModelAsync)

使用内存存储管理下载任务状态。
生产环境建议使用 Redis 或数据库持久化。
"""

from typing import Any

from app.observability.logging import get_logger
from app.services.ollama import DownloadTaskStore, get_ollama_service

logger = get_logger(__name__)

# 全局下载任务存储
_download_store = DownloadTaskStore()


async def start_model_download(model_name: str) -> dict[str, Any]:
    """启动 Ollama 模型下载任务

    对齐 WeKnora99 DownloadOllamaModel

    Args:
        model_name: 模型名称

    Returns:
        任务信息
    """
    ollama = await get_ollama_service()

    # 检查是否已有进行中的任务
    existing = await _download_store.find_by_model(model_name)
    if existing and existing["status"] in ("pending", "downloading"):
        logger.info("ollama_download_task_exists", model_name=model_name)
        return existing

    # 检查模型是否已存在
    if await ollama.is_model_available(model_name):
        return {
            "id": "",
            "modelName": model_name,
            "status": "completed",
            "progress": 100.0,
            "message": "模型已存在",
            "startTime": existing["startTime"] if existing else "",
            "endTime": None,
        }

    # 创建新任务
    task = await _download_store.create(model_name)

    # 启动异步下载
    import asyncio

    asyncio.create_task(_pull_model_async(task["id"], model_name))

    return task


async def _pull_model_async(task_id: str, model_name: str) -> None:
    """异步拉取模型

    对齐 WeKnora99 downloadModelAsync

    Args:
        task_id: 任务 ID
        model_name: 模型名称
    """

    # 更新任务状态为下载中
    await _download_store.update(task_id, status="downloading", progress=0.0)

    def progress_callback(progress: float, message: str) -> None:
        """进度回调"""
        import asyncio

        asyncio.create_task(
            _download_store.update(
                task_id,
                progress=progress,
                message=message,
            )
        )

    # 执行下载
    ollama = await get_ollama_service()
    success = await ollama.pull_model(model_name, progress_callback)

    if success:
        await _download_store.update(
            task_id,
            status="completed",
            progress=100.0,
            message="下载完成",
        )
        logger.info("ollama_download_completed", task_id=task_id, model_name=model_name)
    else:
        await _download_store.update(
            task_id,
            status="failed",
            message="下载失败",
        )
        logger.error("ollama_download_failed", task_id=task_id, model_name=model_name)


async def get_download_task(task_id: str) -> dict[str, Any] | None:
    """获取下载任务信息

    对齐 WeKnora99 GetDownloadProgress

    Args:
        task_id: 任务 ID

    Returns:
        任务信息或 None
    """
    return await _download_store.get(task_id)


async def cancel_download_task(task_id: str) -> bool:
    """取消下载任务

    Args:
        task_id: 任务 ID

    Returns:
        是否取消成功
    """
    task = await _download_store.get(task_id)
    if not task:
        return False

    # 只有 pending 状态可以取消
    if task["status"] != "pending":
        return False

    await _download_store.update(task_id, status="cancelled", message="任务已取消")
    logger.info("ollama_download_cancelled", task_id=task_id)

    return True


async def list_download_tasks() -> list[dict[str, Any]]:
    """列出所有下载任务

    对齐 WeKnora99 ListDownloadTasks

    Returns:
        任务列表
    """
    return await _download_store.list_all()


__all__ = [
    "start_model_download",
    "get_download_task",
    "cancel_download_task",
    "list_download_tasks",
]
