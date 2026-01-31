"""系统初始化服务

提供知识库初始化、模型测试等功能
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.repositories.knowledge import KnowledgeBaseRepository
from app.repositories.model import ModelRepository
from app.schemas.initialization import (
    EmbeddingTestRequest,
    KBModelConfigRequest,
    ModelTestResponse,
    OllamaStatusResponse,
    RemoteModelCheckRequest,
    RerankTestRequest,
)
from app.services.model_test import ModelTestService

logger = get_logger(__name__)


class InitializationService:
    """系统初始化服务"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._kb_repo: KnowledgeBaseRepository | None = None
        self._model_repo: ModelRepository | None = None

    @property
    def kb_repo(self) -> KnowledgeBaseRepository:
        """延迟初始化知识库仓储"""
        if self._kb_repo is None:
            self._kb_repo = KnowledgeBaseRepository(self.session)
        return self._kb_repo

    @property
    def model_repo(self) -> ModelRepository:
        """延迟初始化模型仓储"""
        if self._model_repo is None:
            self._model_repo = ModelRepository(self.session)
        return self._model_repo

    async def get_kb_config(self, kb_id: str, tenant_id: int) -> dict | None:
        """获取知识库配置

        对齐 WeKnora99 GET /initialization/kb/{kbId}/config

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            配置响应
        """
        kb = await self.kb_repo.get_by_tenant(kb_id, tenant_id)
        if not kb:
            return None

        # 获取模型信息
        llm_model = None
        if kb.summary_model_id:
            llm_model = await self.model_repo.get_by_tenant(kb.summary_model_id, tenant_id)

        embedding_model = None
        if kb.embedding_model_id:
            embedding_model = await self.model_repo.get_by_tenant(
                kb.embedding_model_id, tenant_id
            )

        # 检查是否有文件
        has_files = await self.kb_repo.get_knowledge_count(kb_id, tenant_id) > 0

        config = {
            "has_files": has_files,
            "llm": self._build_model_response(llm_model),
            "embedding": self._build_embedding_response(embedding_model),
            "rerank": {"enabled": False},
            "multimodal": self._build_multimodal_response(kb),
            "document_splitting": {
                "chunkSize": kb.chunking_config.get("chunk_size", 1000),
                "chunkOverlap": kb.chunking_config.get("chunk_overlap", 200),
                "separators": kb.chunking_config.get("separators", ["\n\n", "\n", "。"]),
            },
            "node_extract": {"enabled": False},
        }

        return config

    def _build_model_response(self, model: Any | None) -> dict | None:
        """构建 LLM 模型响应"""
        if not model:
            return None

        return {
            "source": model.source,
            "modelName": model.name,
            "baseUrl": model.parameters.get("base_url", ""),
            "apiKey": self._mask_api_key(model.parameters.get("api_key", "")),
        }

    def _build_embedding_response(self, model: Any | None) -> dict | None:
        """构建 Embedding 模型响应"""
        if not model:
            return None

        embedding_params = model.parameters.get("embedding_parameters") or {}
        return {
            "source": model.source,
            "modelName": model.name,
            "baseUrl": model.parameters.get("base_url", ""),
            "apiKey": self._mask_api_key(model.parameters.get("api_key", "")),
            "dimension": embedding_params.get("dimension", 0),
        }

    def _build_multimodal_response(self, kb: Any) -> dict:
        """构建多模态响应"""
        vlm_config = kb.vlm_config or {}
        cos_config = kb.cos_config or {}

        return {
            "enabled": vlm_config.get("enabled", False),
            "storageType": cos_config.get("provider", "cos") if cos_config else "",
        }

    def _mask_api_key(self, api_key: str) -> str:
        """脱敏 API Key"""
        if not api_key:
            return ""
        if len(api_key) <= 8:
            return "***"
        return api_key[:4] + "***" + api_key[-4:]

    async def update_kb_config(
        self, kb_id: str, data: KBModelConfigRequest, tenant_id: int
    ) -> dict | None:
        """更新知识库配置

        对齐 WeKnora99 PUT /initialization/kb/{kbId}/config

        Args:
            kb_id: 知识库 ID
            data: 配置请求
            tenant_id: 租户 ID

        Returns:
            更新后的配置
        """
        kb = await self.kb_repo.get_by_tenant(kb_id, tenant_id)
        if not kb:
            return None

        # 检查 Embedding 模型是否可修改
        if kb.embedding_model_id and kb.embedding_model_id != data.embedding_model_id:
            knowledge_count = await self.kb_repo.get_knowledge_count(kb_id, tenant_id)
            if knowledge_count > 0:
                raise ValueError("知识库中已有文件，无法修改 Embedding 模型")

        # 更新模型 ID
        kb.summary_model_id = data.llm_model_id
        kb.embedding_model_id = data.embedding_model_id

        # 更新 VLM 配置
        if data.vlm_config:
            kb.vlm_config = {
                "enabled": data.vlm_config.enabled,
                "model_id": data.vlm_config.model_name,
            }
        elif not data.multimodal.enabled:
            kb.vlm_config = {"enabled": False, "model_id": ""}

        # 更新文档分块配置
        kb.chunking_config = {
            "chunk_size": data.document_splitting.chunk_size,
            "chunk_overlap": data.document_splitting.chunk_overlap,
            "separators": data.document_splitting.separators,
            "enable_multimodal": data.multimodal.enabled,
        }

        # 更新存储配置
        if data.multimodal.enabled:
            storage_type = data.multimodal.storage_type.lower()
            kb.cos_config = {"provider": storage_type}

        await self.session.commit()

        logger.info(
            "kb_config_updated",
            kb_id=kb_id,
            tenant_id=tenant_id,
        )

        return await self.get_kb_config(kb_id, tenant_id)

    async def check_ollama_status(self) -> OllamaStatusResponse:
        """检查 Ollama 服务状态

        对齐 WeKnora99 GET /initialization/ollama/status

        Returns:
            Ollama 状态响应
        """
        import os

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        available = False
        version = ""

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/api/version")
                if response.status_code == 200:
                    available = True
                    version = response.json().get("version", "")

            logger.info("ollama_status_check", available=available, version=version)

        except Exception as e:
            logger.warning("ollama_status_check_failed", error=str(e))

        return OllamaStatusResponse(
            available=available, version=version, base_url=base_url
        )

    async def test_embedding_model(self, data: EmbeddingTestRequest) -> ModelTestResponse:
        """测试 Embedding 模型

        对齐 WeKnora99 POST /initialization/models/embedding/test

        Args:
            data: 测试请求

        Returns:
            测试结果
        """
        test_service = ModelTestService(self.session)

        return await test_service.test_embedding(
            source=data.source,
            model_name=data.model_name,
            base_url=data.base_url,
            api_key=data.api_key,
            dimension=data.dimension,
            provider=data.provider,
        )

    async def check_rerank_model(self, data: RerankTestRequest) -> ModelTestResponse:
        """测试 Rerank 模型

        对齐 WeKnora99 POST /initialization/models/rerank/check

        Args:
            data: 测试请求

        Returns:
            测试结果
        """
        test_service = ModelTestService(self.session)

        result = await test_service.test_rerank(
            model_name=data.model_name,
            base_url=data.base_url,
            api_key=data.api_key,
            provider="",
        )

        return ModelTestResponse(
            available=result.status.value == "success",
            message=result.message,
        )

    async def check_remote_model(self, data: RemoteModelCheckRequest) -> ModelTestResponse:
        """检查远程模型连接

        对齐 WeKnora99 POST /initialization/models/remote/check

        Args:
            data: 检查请求

        Returns:
            检查结果
        """
        test_service = ModelTestService(self.session)

        # 使用 LLM 测试来检查远程模型
        result = await test_service.test_llm(
            model_name=data.model_name,
            base_url=data.base_url,
            api_key=data.api_key,
            provider="",
        )

        return ModelTestResponse(
            available=result.status.value == "success",
            message=result.message,
        )

    # ============== Ollama 模型管理方法 ==============

    async def list_ollama_models(self) -> dict:
        """列出已安装的 Ollama 模型

        对齐 WeKnora99 GET /initialization/ollama/models

        Returns:
            模型列表响应
        """
        import os

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        models = []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])

            logger.info("ollama_models_listed", count=len(models))

        except Exception as e:
            logger.warning("ollama_models_list_failed", error=str(e))

        return {"models": models}

    async def check_ollama_models(self, model_names: list[str]) -> dict:
        """检查指定的 Ollama 模型是否已安装

        对齐 WeKnora99 POST /initialization/ollama/models/check

        Args:
            model_names: 模型名称列表

        Returns:
            模型状态映射
        """
        import os

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model_status = {}

        try:
            import httpx

            # 获取已安装的模型列表
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    installed_models = {
                        m.get("name", "").split(":")[0]: True
                        for m in data.get("models", [])
                    }

                    # 检查每个请求的模型
                    for model_name in model_names:
                        model_status[model_name] = installed_models.get(
                            model_name.split(":")[0], False
                        )

            logger.info("ollama_models_checked", models=model_status)

        except Exception as e:
            logger.warning("ollama_models_check_failed", error=str(e))
            # 出错时默认所有模型不可用
            model_status = dict.fromkeys(model_names, False)

        return {"models": model_status}

    async def download_ollama_model(self, model_name: str) -> dict:
        """异步下载指定的 Ollama 模型

        对齐 WeKnora99 POST /initialization/ollama/models/download

        Args:
            model_name: 模型名称

        Returns:
            下载任务信息
        """
        import os
        import uuid

        # TODO: 实现实际的异步下载逻辑
        # 这里需要使用后台任务或 WebSocket 来实现进度推送

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # 先检查模型是否已存在
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    installed_models = [
                        m.get("name", "").split(":")[0]
                        for m in data.get("models", [])
                    ]
                    if model_name.split(":")[0] in installed_models:
                        return {
                            "task_id": "",
                            "modelName": model_name,
                            "status": "completed",
                            "progress": 100.0,
                            "message": "模型已存在",
                        }
        except Exception:
            pass

        # 创建下载任务（预留实现）
        task_id = str(uuid.uuid4())

        logger.info(
            "ollama_download_started",
            task_id=task_id,
            model_name=model_name,
        )

        # TODO: 启动后台下载任务
        # asyncio.create_task(self._pull_model_async(task_id, model_name))

        return {
            "task_id": task_id,
            "modelName": model_name,
            "status": "pending",
            "progress": 0.0,
            "message": "下载任务已创建（预留实现）",
        }

    async def get_download_progress(self, task_id: str) -> dict | None:
        """获取 Ollama 模型下载任务的进度

        对齐 WeKnora99 GET /initialization/ollama/download/progress/{taskId}

        Args:
            task_id: 任务 ID

        Returns:
            任务进度信息
        """
        # TODO: 实现实际的下载进度查询
        # 这里需要从任务存储中获取进度信息

        logger.info("ollama_download_progress_queried", task_id=task_id)

        return {
            "id": task_id,
            "modelName": "",
            "status": "pending",
            "progress": 0.0,
            "message": "下载进度查询（预留实现）",
        }

    async def list_download_tasks(self) -> list:
        """列出所有 Ollama 模型下载任务

        对齐 WeKnora99 GET /initialization/ollama/download/tasks

        Returns:
            任务列表
        """
        # TODO: 实现实际的下载任务列表查询
        logger.info("ollama_download_tasks_listed")

        return []


__all__ = ["InitializationService"]
