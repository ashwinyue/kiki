"""Ollama 本地模型管理服务

对齐 WeKnora99 Ollama 服务实现
(internal/models/utils/ollama/ollama.go)

提供以下功能:
- list_models(): 列出本地模型
- pull_model(): 拉取模型
- delete_model(): 删除模型
- get_model_info(): 获取模型信息
- get_download_progress(): 获取下载进度
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 下载任务存储 ==============


class DownloadTaskStore:
    """下载任务内存存储

    使用内存存储下载任务状态。
    生产环境建议使用 Redis 或数据库持久化。
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._model_tasks: dict[str, str] = {}  # model_name -> task_id
        self._lock = asyncio.Lock()

    async def create(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """创建新的下载任务

        Args:
            model_name: 模型名称

        Returns:
            任务信息
        """
        task_id = str(uuid4())
        now = datetime.now(UTC).isoformat()

        task = {
            "id": task_id,
            "modelName": model_name,
            "status": "pending",
            "progress": 0.0,
            "message": "准备下载",
            "startTime": now,
            "endTime": None,
        }

        async with self._lock:
            self._tasks[task_id] = task
            self._model_tasks[model_name] = task_id

        logger.info("ollama_download_task_created", task_id=task_id, model_name=model_name)
        return task

    async def get(self, task_id: str) -> dict[str, Any] | None:
        """获取任务信息

        Args:
            task_id: 任务 ID

        Returns:
            任务信息或 None
        """
        async with self._lock:
            return self._tasks.get(task_id)

    async def update(
        self,
        task_id: str,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
    ) -> dict[str, Any] | None:
        """更新任务状态

        Args:
            task_id: 任务 ID
            status: 新状态
            progress: 进度百分比
            message: 状态消息

        Returns:
            更新后的任务信息或 None
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if status is not None:
                task["status"] = status
                # 完成或失败时记录结束时间
                if status in ("completed", "failed"):
                    task["endTime"] = datetime.now(UTC).isoformat()

            if progress is not None:
                task["progress"] = progress

            if message is not None:
                task["message"] = message

            return task

    async def list_all(self) -> list[dict[str, Any]]:
        """列出所有任务

        Returns:
            任务列表
        """
        async with self._lock:
            return list(self._tasks.values())

    async def find_by_model(self, model_name: str) -> dict[str, Any] | None:
        """根据模型名查找任务

        Args:
            model_name: 模型名称

        Returns:
            任务信息或 None
        """
        async with self._lock:
            task_id = self._model_tasks.get(model_name)
            if task_id:
                return self._tasks.get(task_id)
            return None

    async def delete(self, task_id: str) -> bool:
        """删除任务

        Args:
            task_id: 任务 ID

        Returns:
            是否删除成功
        """
        async with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                self._model_tasks.pop(task.get("modelName", ""), None)
                return True
            return False


# 全局任务存储实例
_download_store = DownloadTaskStore()


# ============== Ollama 服务 ==============


class OllamaService:
    """Ollama 本地模型管理服务

    对齐 WeKnora99 OllamaService
    (internal/models/utils/ollama/ollama.go)

    使用 Ollama HTTP API 管理本地模型。
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        """初始化 Ollama 服务

        Args:
            base_url: Ollama 服务地址，默认从环境变量读取
            timeout: 请求超时时间（秒）
        """
        import os

        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._timeout = timeout
        self._is_available: bool | None = None
        self._version: str | None = None

    @property
    def base_url(self) -> str:
        """获取 Ollama 服务地址"""
        return self._base_url

    async def is_available(self) -> bool:
        """检查 Ollama 服务是否可用

        Returns:
            服务是否可用
        """
        if self._is_available is not None:
            return self._is_available

        try:
            version = await self.get_version()
            self._is_available = version != ""
            return self._is_available
        except Exception:
            self._is_available = False
            return False

    async def get_version(self) -> str:
        """获取 Ollama 版本

        Returns:
            版本字符串
        """
        if self._version:
            return self._version

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/version")
                if response.status_code == 200:
                    data = response.json()
                    self._version = data.get("version", "")
                    return self._version
        except Exception as e:
            logger.warning("ollama_version_check_failed", error=str(e))

        return ""

    async def list_models(self) -> list[dict[str, Any]]:
        """列出已安装的模型

        对齐 WeKnora99 ListModelsDetailed

        Returns:
            模型列表，包含模型详细信息
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])

                    logger.info("ollama_models_listed", count=len(models))
                    return models

        except Exception as e:
            logger.warning("ollama_list_models_failed", error=str(e))

        return []

    async def is_model_available(self, model_name: str) -> bool:
        """检查模型是否已安装

        Args:
            model_name: 模型名称

        Returns:
            模型是否可用
        """
        # 标准化模型名称（添加 :latest 如果没有版本）
        check_name = model_name
        if ":" not in model_name:
            check_name = f"{model_name}:latest"

        models = await self.list_models()
        for model in models:
            if model.get("name") == check_name:
                return True

        return False

    async def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """获取模型详细信息

        Args:
            model_name: 模型名称

        Returns:
            模型信息或 None
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model_name},
                )
                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.warning("ollama_model_info_failed", model_name=model_name, error=str(e))

        return None

    async def pull_model(
        self,
        model_name: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """拉取模型

        对齐 WeKnora99 PullModel

        Args:
            model_name: 模型名称
            progress_callback: 进度回调函数 (progress: float, message: str) -> None

        Returns:
            是否成功
        """
        # 检查模型是否已存在
        if await self.is_model_available(model_name):
            logger.info("ollama_model_already_exists", model_name=model_name)
            if progress_callback:
                progress_callback(100.0, "模型已存在")
            return True

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # 使用流式响应获取进度
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        return False

                    # 读取流式响应
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            import json

                            data = json.loads(line)
                            progress = 0.0
                            message = "下载中"

                            # 解析进度信息
                            if "total" in data and "completed" in data:
                                total = data.get("total", 0)
                                completed = data.get("completed", 0)
                                if total > 0:
                                    progress = (completed / total) * 100

                            status = data.get("status", "")
                            if status:
                                message = status

                            if progress_callback:
                                progress_callback(progress, message)

                            logger.debug(
                                "ollama_pull_progress",
                                model_name=model_name,
                                progress=progress,
                                message=message,
                            )

                        except (json.JSONDecodeError, KeyError):
                            pass

            logger.info("ollama_model_pulled", model_name=model_name)
            return True

        except Exception as e:
            logger.error("ollama_pull_failed", model_name=model_name, error=str(e))
            return False

    async def delete_model(self, model_name: str) -> bool:
        """删除模型

        对齐 WeKnora99 DeleteModel

        Args:
            model_name: 模型名称

        Returns:
            是否成功
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self._base_url}/api/delete",
                    params={"name": model_name},
                )
                if response.status_code == 200:
                    logger.info("ollama_model_deleted", model_name=model_name)
                    return True

        except Exception as e:
            logger.warning("ollama_delete_failed", model_name=model_name, error=str(e))

        return False

    async def create_download_task(self, model_name: str) -> dict[str, Any]:
        """创建模型下载任务

        对齐 WeKnora99 DownloadOllamaModel

        Args:
            model_name: 模型名称

        Returns:
            任务信息
        """
        # 检查是否已有进行中的任务
        existing = await _download_store.find_by_model(model_name)
        if existing and existing["status"] in ("pending", "downloading"):
            logger.info("ollama_download_task_exists", model_name=model_name)
            return existing

        # 检查模型是否已存在
        if await self.is_model_available(model_name):
            return {
                "id": "",
                "modelName": model_name,
                "status": "completed",
                "progress": 100.0,
                "message": "模型已存在",
                "startTime": datetime.now(UTC).isoformat(),
                "endTime": None,
            }

        # 创建新任务
        task = await _download_store.create(model_name)

        # 启动异步下载
        asyncio.create_task(self._pull_model_async(task["id"], model_name))

        return task

    async def _pull_model_async(self, task_id: str, model_name: str) -> None:
        """异步拉取模型

        Args:
            task_id: 任务 ID
            model_name: 模型名称
        """
        # 更新任务状态为下载中
        await _download_store.update(task_id, status="downloading", progress=0.0)

        def progress_callback(progress: float, message: str) -> None:
            """进度回调"""
            asyncio.create_task(
                _download_store.update(
                    task_id,
                    progress=progress,
                    message=message,
                )
            )

        # 执行下载
        success = await self.pull_model(model_name, progress_callback)

        if success:
            await _download_store.update(task_id, status="completed", progress=100.0)
            logger.info("ollama_download_completed", task_id=task_id, model_name=model_name)
        else:
            await _download_store.update(
                task_id, status="failed", message="下载失败"
            )
            logger.error("ollama_download_failed", task_id=task_id, model_name=model_name)

    async def get_download_progress(self, task_id: str) -> dict[str, Any] | None:
        """获取下载任务进度

        对齐 WeKnora99 GetDownloadProgress

        Args:
            task_id: 任务 ID

        Returns:
            任务进度信息或 None
        """
        return await _download_store.get(task_id)

    async def list_download_tasks(self) -> list[dict[str, Any]]:
        """列出所有下载任务

        对齐 WeKnora99 ListDownloadTasks

        Returns:
            任务列表
        """
        return await _download_store.list_all()


# ============== 单例实例 ==============

_ollama_service: OllamaService | None = None
_service_lock = asyncio.Lock()


async def get_ollama_service() -> OllamaService:
    """获取 Ollama 服务单例

    Returns:
        OllamaService 实例
    """
    global _ollama_service

    if _ollama_service is not None:
        return _ollama_service

    async with _service_lock:
        if _ollama_service is None:
            _ollama_service = OllamaService()

    return _ollama_service


__all__ = [
    "OllamaService",
    "get_ollama_service",
    "DownloadTaskStore",
]
