"""Pinecone 向量存储

支持 Pinecone 云端向量搜索服务。
文档: https://docs.pinecone.io/

依赖安装:
    uv add -E pinecone pinecone
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.observability.logging import get_logger
from app.vector_stores.base import (
    BaseVectorStore,
    IndexResult,
    SearchResult,
    VectorStats,
    VectorStoreConfig,
)

logger = get_logger(__name__)


@dataclass
class PineconeConfig(VectorStoreConfig):
    """Pinecone 配置

    Attributes:
        api_key: Pinecone API Key
        environment: 环境 (gcp-starter, aws-us-east-1, etc.)
        project_id: 项目 ID（新版 API）
        region: 区域（新版 API）
        cloud: 云平台 (aws, gcp, azure)
        host: 自定义主机地址
        namespace: 命名空间（可选，用于多租户）
    """

    api_key: str | None = None
    environment: str | None = None
    project_id: str | None = None
    region: str = "us-east-1"
    cloud: str = "aws"
    host: str | None = None
    namespace: str = ""


class PineconeVectorStore(BaseVectorStore):
    """Pinecone 向量存储

    支持 Pinecone 云端服务。
    """

    def __init__(
        self,
        config: PineconeConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化 Pinecone 向量存储

        Args:
            config: Pinecone 配置
            embeddings: Embedding 实例
        """
        super().__init__(config, embeddings)
        self.pinecone_config: PineconeConfig = config or PineconeConfig()
        self._client: Any = None
        self._index: Any = None

    async def initialize(self) -> None:
        """初始化 Pinecone 客户端"""
        try:
            from pinecone import Pinecone, ServerlessSpec

            api_key = self.pinecone_config.api_key
            if not api_key:
                raise ValueError("Pinecone API Key is required")

            # 创建客户端
            self._client = Pinecone(api_key=api_key)

            # 检查/创建索引
            existing_indexes = [idx.name for idx in self._client.list_indexes()]

            if self.config.collection_name not in existing_indexes:
                # 创建索引
                self._client.create_index(
                    name=self.config.collection_name,
                    dimension=self.config.dimension,
                    metric=self._get_metric(),
                    spec=ServerlessSpec(
                        cloud=self.pinecone_config.cloud,
                        region=self.pinecone_config.region,
                    ),
                )
                logger.info(
                    "pinecone_index_created",
                    index=self.config.collection_name,
                    dimension=self.config.dimension,
                )

            # 获取索引
            self._index = self._client.Index(self.config.collection_name)

            self._initialized = True
            logger.info(
                "pinecone_initialized",
                index=self.config.collection_name,
                region=self.pinecone_config.region,
            )

        except ImportError as e:
            logger.error("pinecone_not_installed")
            raise ImportError(
                "请安装 pinecone: uv add -E pinecone pinecone"
            ) from e
        except Exception as e:
            logger.error("pinecone_init_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if self._index is None:
                return False
            # 尝试查询统计信息
            self._index.describe_index_stats()
            return True
        except Exception as e:
            logger.warning("pinecone_health_check_failed", error=str(e))
            return False

    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文档"""
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        import time

        # 生成 ID
        if ids is None:
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 嵌入文档
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)

        # 准备 Pinecone 记录
        records = [
            {
                "id": doc_id,
                "values": vector,
                "metadata": {
                    "text": doc.page_content,
                    **doc.metadata,
                    "tenant_id": self.config.tenant_id or 0,
                },
            }
            for doc_id, vector, doc in zip(ids, vectors, documents, strict=True)
        ]

        # 命名空间处理
        namespace = self._get_namespace()

        # 上传（使用 asyncio.to_thread 处理同步操作）
        import asyncio

        await asyncio.to_thread(
            self._index.upsert,
            vectors=records,
            namespace=namespace,
        )

        logger.info(
            "pinecone_documents_added",
            count=len(documents),
            index=self.config.collection_name,
        )

        return IndexResult(ids=ids, count=len(documents))

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文本"""
        documents = [
            Document(
                page_content=text,
                metadata=(metadatas or [{}])[i] if metadatas else {},
            )
            for i, text in enumerate(texts)
        ]
        return await self.add_documents(documents, ids)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """相似度搜索"""
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        # 嵌入查询
        query_vector = await self.embeddings.aembed_query(query)

        return await self.search_by_vector(query_vector, k, score_threshold, filter_dict)

    async def search_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """通过向量搜索"""
        await self.ensure_initialized()

        import asyncio

        # 构建过滤条件
        query_filter = None
        if filter_dict or self.config.tenant_id:
            filter_dict = filter_dict or {}
            if self.config.tenant_id:
                filter_dict["tenant_id"] = self.config.tenant_id
            query_filter = filter_dict

        # 命名空间
        namespace = self._get_namespace()

        # 搜索
        response = await asyncio.to_thread(
            self._index.query,
            vector=vector,
            top_k=k,
            filter=query_filter,
            namespace=namespace,
            include_metadata=True,
        )

        # 转换结果
        search_results = []
        for match in response.matches:
            # Pinecone 返回的分数是相似度，需要根据 metric 转换
            score = match.score
            if self.config.metric == "euclidean":
                # 欧式距离越小越相似，需要反转
                score = 1 / (1 + score)

            if score_threshold is None or score >= score_threshold:
                metadata = match.metadata or {}
                search_results.append(
                    SearchResult(
                        content=metadata.pop("text", ""),
                        metadata=metadata,
                        score=score,
                        id=match.id,
                    )
                )

        return search_results

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.ensure_initialized()

        try:
            import asyncio

            namespace = self._get_namespace()

            await asyncio.to_thread(
                self._index.delete,
                ids=ids,
                namespace=namespace,
            )

            logger.info(
                "pinecone_documents_deleted",
                count=len(ids),
                index=self.config.collection_name,
            )
            return True

        except Exception as e:
            logger.error("pinecone_delete_failed", error=str(e))
            return False

    async def delete_collection(self) -> bool:
        """删除整个集合"""
        await self.ensure_initialized()

        try:
            import asyncio

            await asyncio.to_thread(
                self._client.delete_index,
                self.config.collection_name,
            )

            logger.info(
                "pinecone_index_deleted",
                index=self.config.collection_name,
            )
            return True

        except Exception as e:
            logger.error("pinecone_delete_index_failed", error=str(e))
            return False

    async def get_stats(self) -> VectorStats:
        """获取统计信息"""
        await self.ensure_initialized()

        try:
            import asyncio

            stats = await asyncio.to_thread(self._index.describe_index_stats)

            return VectorStats(
                total_vectors=stats.total_vector_count or 0,
                collections=1,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )
        except Exception as e:
            logger.error("pinecone_get_stats_failed", error=str(e))
            return VectorStats(
                total_vectors=0,
                collections=0,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )

    def _get_namespace(self) -> str:
        """获取命名空间"""
        # 使用租户 ID 作为命名空间，实现多租户隔离
        if self.config.tenant_id:
            return f"tenant_{self.config.tenant_id}"
        return self.pinecone_config.namespace or ""

    def _get_metric(self) -> str:
        """获取距离度量"""
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dotproduct": "dotproduct",
        }
        return metric_map.get(self.config.metric, "cosine")


__all__ = [
    "PineconeConfig",
    "PineconeVectorStore",
]
