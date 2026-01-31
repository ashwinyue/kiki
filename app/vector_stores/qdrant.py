"""Qdrant 向量存储

支持本地和远程 Qdrant 实例。
文档: https://qdrant.tech/documentation/

依赖安装:
    uv add qdrant-client
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
class QdrantConfig(VectorStoreConfig):
    """Qdrant 配置

    Attributes:
        url: Qdrant 服务器 URL
        api_key: API Key（云端模式需要）
        location: 部署位置 (local, cloud)
        path: 本地存储路径（本地模式）
        port: 本地模式端口
        grpc_port: GRPC 端口
        prefer_grpc: 是否优先使用 GRPC
        timeout: 请求超时时间（秒）
    """

    url: str | None = None
    api_key: str | None = None
    location: str = "local"  # local, cloud
    path: str = "./data/qdrant"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    timeout: int = 60


class QdrantVectorStore(BaseVectorStore):
    """Qdrant 向量存储

    支持本地和云端部署。
    """

    def __init__(
        self,
        config: QdrantConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化 Qdrant 向量存储

        Args:
            config: Qdrant 配置
            embeddings: Embedding 实例
        """
        super().__init__(config, embeddings)
        self.qdrant_config: QdrantConfig = config or QdrantConfig()
        self._client: Any = None

    async def initialize(self) -> None:
        """初始化 Qdrant 客户端"""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import VectorParams

            # 创建客户端
            if self.qdrant_config.location == "cloud":
                if not self.qdrant_config.url or not self.qdrant_config.api_key:
                    raise ValueError("Cloud mode requires url and api_key")

                self._client = AsyncQdrantClient(
                    url=self.qdrant_config.url,
                    api_key=self.qdrant_config.api_key,
                    timeout=self.qdrant_config.timeout,
                )
            else:
                self._client = AsyncQdrantClient(
                    path=self.qdrant_config.path,
                    port=self.qdrant_config.port,
                    timeout=self.qdrant_config.timeout,
                )

            # 检查/创建集合
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name not in collection_names:
                # 创建集合
                await self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=self._get_distance(),
                    ),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self.config.collection_name,
                    dimension=self.config.dimension,
                )

            self._initialized = True
            logger.info(
                "qdrant_initialized",
                collection=self.config.collection_name,
                location=self.qdrant_config.location,
            )

        except ImportError as e:
            logger.error("qdrant_client_not_installed")
            raise ImportError(
                "请安装 qdrant-client: uv add qdrant-client"
            ) from e
        except Exception as e:
            logger.error("qdrant_init_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if self._client is None:
                return False
            # 尝试获取集合列表
            await self._client.get_collections()
            return True
        except Exception as e:
            logger.warning("qdrant_health_check_failed", error=str(e))
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

        from qdrant_client.models import PointStruct

        # 生成 ID
        if ids is None:
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 嵌入文档
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)

        # 准备 Qdrant Points
        points = [
            PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "tenant_id": self.config.tenant_id,
                },
            )
            for doc_id, vector, doc in zip(ids, vectors, documents, strict=True)
        ]

        # 上传
        await self._client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

        logger.info(
            "qdrant_documents_added",
            count=len(documents),
            collection=self.config.collection_name,
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

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # 构建过滤条件
        query_filter = None
        if filter_dict or self.config.tenant_id:
            conditions = []

            # 租户隔离
            if self.config.tenant_id:
                conditions.append(
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=self.config.tenant_id),
                    )
                )

            # 额外过滤
            if filter_dict:
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value),
                        )
                    )

            if conditions:
                query_filter = Filter(must=conditions)

        # 搜索

        results = await self._client.search(
            collection_name=self.config.collection_name,
            query_vector=vector,
            limit=k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # 转换结果
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(
                SearchResult(
                    content=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    score=result.score,
                    id=str(result.id),
                )
            )

        return search_results

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.ensure_initialized()

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            await self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="id",
                            match=MatchAny(any=ids),
                        )
                    ]
                ),
            )

            logger.info(
                "qdrant_documents_deleted",
                count=len(ids),
                collection=self.config.collection_name,
            )
            return True

        except Exception as e:
            logger.error("qdrant_delete_failed", error=str(e))
            return False

    async def delete_collection(self) -> bool:
        """删除整个集合"""
        await self.ensure_initialized()

        try:
            await self._client.delete_collection(self.config.collection_name)
            logger.info(
                "qdrant_collection_deleted",
                collection=self.config.collection_name,
            )
            return True
        except Exception as e:
            logger.error("qdrant_delete_collection_failed", error=str(e))
            return False

    async def get_stats(self) -> VectorStats:
        """获取统计信息"""
        await self.ensure_initialized()

        try:
            collection_info = await self._client.get_collection(
                self.config.collection_name
            )
            config = collection_info.config.params

            return VectorStats(
                total_vectors=collection_info.points_count or 0,
                collections=1,
                dimension=config.vectors.size if config.vectors else self.config.dimension,
                metric=config.vectors.distance.value if config.vectors else self.config.metric,
            )
        except Exception as e:
            logger.error("qdrant_get_stats_failed", error=str(e))
            return VectorStats(
                total_vectors=0,
                collections=0,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )

    def _get_distance(self) -> Any:
        """获取距离度量"""
        from qdrant_client.models import Distance

        metric_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dotproduct": Distance.DOT,
        }
        return metric_map.get(self.config.metric, Distance.COSINE)


__all__ = [
    "QdrantConfig",
    "QdrantVectorStore",
]
