"""知识库初始化服务

对齐 WeKnora99 internal/handler/initialization.go

提供知识库初始化配置相关功能：
- 获取初始化配置
- 更新初始化配置
- 验证配置
- 执行初始化
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeBase
from app.observability.logging import get_logger
from app.repositories.knowledge import KnowledgeBaseRepository
from app.schemas.knowledge_initialization import (
    EmbeddingConfig,
    ExtractConfig,
    InitializationConfig,
    InitializationStatus,
    LLMConfig,
    MinioConfig,
    MultimodalConfig,
    QuestionGenerationConfig,
    RerankConfig,
    StorageConfig,
    ValidationResult,
    VectorStoreConfig,
)

logger = get_logger(__name__)


@dataclass
class InitResult:
    """初始化结果"""

    success: bool
    message: str
    status: InitializationStatus
    progress_percent: float
    error: str | None = None


class KnowledgeInitializationService:
    """知识库初始化服务

    对齐 WeKnora99 InitializationHandler
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._kb_repo: KnowledgeBaseRepository | None = None

    @property
    def kb_repo(self) -> KnowledgeBaseRepository:
        """延迟初始化知识库仓储"""
        if self._kb_repo is None:
            self._kb_repo = KnowledgeBaseRepository(self.session)
        return self._kb_repo

    async def get_config(
        self, kb_id: str, tenant_id: int
    ) -> InitializationConfig | None:
        """获取知识库初始化配置

        对齐 WeKnora99 GET /initialization/kb/{kbId}/config

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            初始化配置，如果知识库不存在则返回 None
        """
        kb = await self.kb_repo.get_by_tenant(kb_id, tenant_id)
        if not kb:
            return None

        # 检查知识库是否有文件
        has_files = await self._has_files(kb_id, tenant_id)

        # 获取模型配置
        llm_config = await self._get_llm_config(kb)
        embedding_config = await self._get_embedding_config(kb)
        rerank_config = await self._get_rerank_config(kb)

        # 获取多模态配置
        multimodal_config = self._get_multimodal_config(kb)

        # 获取知识图谱提取配置
        extract_config = self._get_extract_config(kb)

        # 获取问题生成配置
        question_config = self._get_question_config(kb)

        # 获取向量存储配置
        vector_config = self._get_vector_config(kb)

        return InitializationConfig(
            kb_id=kb.id,
            kb_name=kb.name,
            has_files=has_files,
            llm=llm_config,
            embedding=embedding_config,
            rerank=rerank_config,
            multimodal=multimodal_config,
            extract=extract_config,
            question_generation=question_config,
            vector_store=vector_config,
            chunking={
                "chunk_size": kb.chunking_config.get("chunk_size", 1000),
                "chunk_overlap": kb.chunking_config.get("chunk_overlap", 200),
                "separators": kb.chunking_config.get("separators", ["\n\n", "\n", "。", "."]),
            },
        )

    async def update_config(
        self,
        kb_id: str,
        tenant_id: int,
        config: InitializationConfig,
    ) -> InitResult:
        """更新知识库初始化配置

        对齐 WeKnora99 PUT /initialization/kb/{kbId}/config

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            config: 初始化配置

        Returns:
            初始化结果
        """
        kb = await self.kb_repo.get_by_tenant(kb_id, tenant_id)
        if not kb:
            return InitResult(
                success=False,
                message="知识库不存在",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error="KNOWLEDGE_BASE_NOT_FOUND",
            )

        # 验证配置
        validation = await self.validate_config(kb, config)
        if not validation.valid:
            return InitResult(
                success=False,
                message="配置验证失败",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error=validation.errors[0] if validation.errors else "UNKNOWN_ERROR",
            )

        # 检查是否可以修改 embedding 模型
        if config.embedding and kb.embedding_model_id != config.embedding.model_id:
            if await self._has_files(kb_id, tenant_id):
                return InitResult(
                    success=False,
                    message="知识库中已有文件，无法修改 Embedding 模型",
                    status=InitializationStatus.FAILED,
                    progress_percent=0.0,
                    error="EMBEDDING_MODEL_CHANGE_NOT_ALLOWED",
                )

        # 更新配置
        try:
            await self._apply_config(kb, config)
            kb.updated_at = datetime.now(UTC)
            await self.session.commit()

            logger.info(
                "knowledge_base_config_updated",
                kb_id=kb_id,
                tenant_id=tenant_id,
            )

            return InitResult(
                success=True,
                message="配置更新成功",
                status=InitializationStatus.COMPLETED,
                progress_percent=100.0,
            )

        except Exception as e:
            logger.error(
                "knowledge_base_config_update_failed",
                kb_id=kb_id,
                tenant_id=tenant_id,
                error=str(e),
                exc_info=True,
            )
            return InitResult(
                success=False,
                message=f"配置更新失败: {str(e)}",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error=str(e),
            )

    async def validate_config(
        self,
        kb: KnowledgeBase,
        config: InitializationConfig,
    ) -> ValidationResult:
        """验证初始化配置

        对齐 WeKnora99 的验证逻辑

        Args:
            kb: 知识库实例
            config: 初始化配置

        Returns:
            验证结果
        """
        errors: list[str] = []

        # 验证 LLM 配置
        if config.llm:
            llm_error = self._validate_llm_config(config.llm)
            if llm_error:
                errors.append(llm_error)

        # 验证 Embedding 配置
        if config.embedding:
            embedding_error = await self._validate_embedding_config(config.embedding)
            if embedding_error:
                errors.append(embedding_error)

        # 验证 Rerank 配置
        if config.rerank and config.rerank.enabled:
            rerank_error = self._validate_rerank_config(config.rerank)
            if rerank_error:
                errors.append(rerank_error)

        # 验证多模态配置
        if config.multimodal and config.multimodal.enabled:
            multimodal_error = self._validate_multimodal_config(config.multimodal)
            if multimodal_error:
                errors.append(multimodal_error)

        # 验证知识图谱提取配置
        if config.extract and config.extract.enabled:
            extract_error = self._validate_extract_config(config.extract)
            if extract_error:
                errors.append(extract_error)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    async def initialize_kb(
        self,
        kb_id: str,
        tenant_id: int,
        config: InitializationConfig,
    ) -> InitResult:
        """执行知识库初始化

        对齐 WeKnora99 POST /initialization/kb/{kbId}

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            config: 初始化配置

        Returns:
            初始化结果
        """
        kb = await self.kb_repo.get_by_tenant(kb_id, tenant_id)
        if not kb:
            return InitResult(
                success=False,
                message="知识库不存在",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error="KNOWLEDGE_BASE_NOT_FOUND",
            )

        # 验证配置
        validation = await self.validate_config(kb, config)
        if not validation.valid:
            return InitResult(
                success=False,
                message=f"配置验证失败: {validation.errors[0]}",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error=validation.errors[0] if validation.errors else "UNKNOWN_ERROR",
            )

        try:
            # 更新配置
            await self._apply_config(kb, config)
            kb.updated_at = datetime.now(UTC)
            await self.session.commit()

            # 执行初始化任务（创建索引、测试连接等）
            await self._run_initialization_tasks(kb, config)

            logger.info(
                "knowledge_base_initialized",
                kb_id=kb_id,
                tenant_id=tenant_id,
            )

            return InitResult(
                success=True,
                message="知识库初始化成功",
                status=InitializationStatus.COMPLETED,
                progress_percent=100.0,
            )

        except Exception as e:
            logger.error(
                "knowledge_base_initialization_failed",
                kb_id=kb_id,
                tenant_id=tenant_id,
                error=str(e),
                exc_info=True,
            )
            return InitResult(
                success=False,
                message=f"初始化失败: {str(e)}",
                status=InitializationStatus.FAILED,
                progress_percent=0.0,
                error=str(e),
            )

    async def _has_files(self, kb_id: str, tenant_id: int) -> bool:
        """检查知识库是否有文件"""
        count = await self.kb_repo.get_knowledge_count(kb_id, tenant_id)
        return count > 0

    async def _get_llm_config(self, kb: KnowledgeBase) -> LLMConfig | None:
        """获取 LLM 配置"""
        if not kb.summary_model_id:
            return None

        # 从模型仓储获取模型信息（简化实现）
        return LLMConfig(
            model_id=kb.summary_model_id,
            source="",
            model_name="",
            base_url="",
            api_key="",
        )

    async def _get_embedding_config(
        self, kb: KnowledgeBase
    ) -> EmbeddingConfig | None:
        """获取 Embedding 配置"""
        if not kb.embedding_model_id:
            return None

        return EmbeddingConfig(
            model_id=kb.embedding_model_id,
            source="",
            model_name="",
            base_url="",
            api_key="",
            dimension=0,
        )

    async def _get_rerank_config(self, kb: KnowledgeBase) -> RerankConfig:
        """获取 Rerank 配置"""
        if not kb.rerank_model_id:
            return RerankConfig(enabled=False)

        return RerankConfig(
            enabled=True,
            model_id=kb.rerank_model_id,
            source="",
            model_name="",
            base_url="",
            api_key="",
        )

    def _get_multimodal_config(self, kb: KnowledgeBase) -> MultimodalConfig:
        """获取多模态配置"""
        vlm_config = kb.vlm_config or {}
        storage_config = kb.cos_config or {}

        return MultimodalConfig(
            enabled=bool(vlm_config.get("enabled", False)),
            vlm_model_id=vlm_config.get("model_id", ""),
            storage_type=storage_config.get("provider", "cos"),
            cos=StorageConfig(
                secret_id=storage_config.get("secret_id", ""),
                secret_key=storage_config.get("secret_key", ""),
                region=storage_config.get("region", ""),
                bucket_name=storage_config.get("bucket_name", ""),
                app_id=storage_config.get("app_id", ""),
                path_prefix=storage_config.get("path_prefix", ""),
            ) if storage_config.get("provider") == "cos" else None,
            minio=MinioConfig(
                bucket_name=storage_config.get("bucket_name", ""),
                path_prefix=storage_config.get("path_prefix", ""),
            ) if storage_config.get("provider") == "minio" else None,
        )

    def _get_extract_config(self, kb: KnowledgeBase) -> ExtractConfig:
        """获取知识图谱提取配置"""
        extract_config = kb.extract_config or {}
        return ExtractConfig(
            enabled=extract_config.get("enabled", False),
            text=extract_config.get("text", ""),
            tags=extract_config.get("tags", []),
            nodes=extract_config.get("nodes", []),
            relations=extract_config.get("relations", []),
        )

    def _get_question_config(
        self, kb: KnowledgeBase
    ) -> QuestionGenerationConfig:
        """获取问题生成配置"""
        config = kb.question_generation_config or {}
        return QuestionGenerationConfig(
            enabled=config.get("enabled", False),
            question_count=config.get("question_count", 3),
        )

    def _get_vector_config(self, kb: KnowledgeBase) -> VectorStoreConfig:
        """获取向量存储配置"""
        # 简化实现，返回默认配置
        return VectorStoreConfig(
            provider="pgvector",
            index_type="hnsw",
            dimension=1536,
        )

    def _validate_llm_config(self, config: LLMConfig) -> str | None:
        """验证 LLM 配置"""
        if not config.model_name:
            return "LLM 模型名称不能为空"
        return None

    async def _validate_embedding_config(
        self, config: EmbeddingConfig
    ) -> str | None:
        """验证 Embedding 配置"""
        if not config.model_name:
            return "Embedding 模型名称不能为空"
        return None

    def _validate_rerank_config(self, config: RerankConfig) -> str | None:
        """验证 Rerank 配置"""
        if not config.model_name:
            return "Rerank 模型名称不能为空"
        if not config.base_url:
            return "Rerank Base URL 不能为空"
        return None

    def _validate_multimodal_config(self, config: MultimodalConfig) -> str | None:
        """验证多模态配置"""
        if not config.vlm_model_id:
            return "启用多模态时需要配置 VLM 模型"

        storage_type = config.storage_type.lower()
        if storage_type == "cos":
            if not config.cos or not config.cos.secret_id:
                return "COS 配置不完整"
        elif storage_type == "minio":
            if not config.minio or not config.minio.bucket_name:
                return "MinIO 配置不完整"

        return None

    def _validate_extract_config(self, config: ExtractConfig) -> str | None:
        """验证知识图谱提取配置"""
        if not config.text:
            return "提取配置中文本不能为空"
        if not config.tags:
            return "提取配置中标签不能为空"
        if not config.nodes or not config.relations:
            return "请先提取实体和关系"
        return None

    async def _apply_config(
        self,
        kb: KnowledgeBase,
        config: InitializationConfig,
    ) -> None:
        """应用配置到知识库"""
        # 更新模型 ID
        if config.llm and config.llm.model_id:
            kb.summary_model_id = config.llm.model_id

        if config.embedding and config.embedding.model_id:
            kb.embedding_model_id = config.embedding.model_id

        if config.rerank:
            if config.rerank.enabled and config.rerank.model_id:
                kb.rerank_model_id = config.rerank.model_id
            else:
                kb.rerank_model_id = None

        # 更新分块配置
        if config.chunking:
            kb.chunking_config = {
                "chunk_size": config.chunking.get("chunk_size", 1000),
                "chunk_overlap": config.chunking.get("chunk_overlap", 200),
                "separators": config.chunking.get("separators", ["\n\n", "\n", "。", "."]),
            }

        # 更新多模态配置
        if config.multimodal:
            if config.multimodal.enabled:
                kb.vlm_config = {
                    "enabled": True,
                    "model_id": config.multimodal.vlm_model_id,
                }

                storage_type = config.multimodal.storage_type.lower()
                if storage_type == "cos" and config.multimodal.cos:
                    kb.cos_config = {
                        "provider": "cos",
                        "secret_id": config.multimodal.cos.secret_id,
                        "secret_key": config.multimodal.cos.secret_key,
                        "region": config.multimodal.cos.region,
                        "bucket_name": config.multimodal.cos.bucket_name,
                        "app_id": config.multimodal.cos.app_id,
                        "path_prefix": config.multimodal.cos.path_prefix,
                    }
                elif storage_type == "minio" and config.multimodal.minio:
                    kb.cos_config = {
                        "provider": "minio",
                        "bucket_name": config.multimodal.minio.bucket_name,
                        "path_prefix": config.multimodal.minio.path_prefix,
                    }
            else:
                kb.vlm_config = {}
                kb.cos_config = {}

        # 更新知识图谱提取配置
        if config.extract:
            kb.extract_config = {
                "enabled": config.extract.enabled,
                "text": config.extract.text,
                "tags": config.extract.tags,
                "nodes": config.extract.nodes,
                "relations": config.extract.relations,
            }

        # 更新问题生成配置
        if config.question_generation:
            kb.question_generation_config = {
                "enabled": config.question_generation.enabled,
                "question_count": config.question_generation.question_count,
            }

    async def _run_initialization_tasks(
        self,
        kb: KnowledgeBase,
        config: InitializationConfig,
    ) -> None:
        """执行初始化任务

        包括：
        - 创建向量索引
        - 测试模型连接
        - 验证存储配置
        """
        # TODO: 实现具体的初始化任务
        # 1. 测试 Embedding 模型连接
        # 2. 创建向量索引（如果需要）
        # 3. 测试存储连接
        # 4. 验证配置完整性

        logger.info(
            "initialization_tasks_started",
            kb_id=kb.id,
            tasks=[
                "test_embedding_connection",
                "create_vector_index",
                "test_storage_connection",
            ],
        )


__all__ = [
    "KnowledgeInitializationService",
    "InitResult",
]
