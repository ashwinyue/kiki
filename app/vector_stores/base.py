"""向量存储抽象基类

定义向量存储的通用接口和配置。
兼容 LangChain VectorStore 接口，提供企业级增强功能。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorStoreConfig:
    """向量存储配置

    Attributes:
        collection_name: 集合/索引名称
        dimension: 向量维度
        metric: 相似度度量方式 (cosine, euclidean, dotproduct)
        embedding_model: Embedding 模型名称
        tenant_id: 租户 ID（用于多租户隔离）
    """

    collection_name: str = "default"
    dimension: int = 1024
    metric: str = "cosine"  # cosine, euclidean, dotproduct
    embedding_model: str = "text-embedding-v4"
    tenant_id: int | None = None

    # 可选的额外配置
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """向量搜索结果

    Attributes:
        content: 文档内容
        metadata: 元数据
        score: 相似度得分 (0-1)
        id: 文档 ID
    """

    content: str
    metadata: dict[str, Any]
    score: float
    id: str | None = None

    def to_document(self) -> Document:
        """转换为 LangChain Document

        Returns:
            Document 实例
        """
        return Document(page_content=self.content, metadata=self.metadata)


@dataclass
class IndexResult:
    """索引结果

    Attributes:
        ids: 文档 ID 列表
        count: 成功索引的文档数量
        failed: 失败的文档数量
    """

    ids: list[str]
    count: int
    failed: int = 0


@dataclass
class VectorStats:
    """向量存储统计信息

    Attributes:
        total_vectors: 向量总数
        collections: 集合数量
        dimension: 向量维度
        metric: 相似度度量方式
    """

    total_vectors: int
    collections: int
    dimension: int
    metric: str


class BaseVectorStore(ABC):
    """向量存储抽象基类

    定义向量存储的通用接口，所有实现必须遵循。
    兼容 LangChain VectorStore 接口，提供企业级增强功能（如多租户隔离）。
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化向量存储

        Args:
            config: 向量存储配置
            embeddings: Embedding 实例
        """
        self.config = config or VectorStoreConfig()
        self.embeddings = embeddings
        self._initialized = False
        self._langchain_store: LangChainVectorStore | None = None

    @abstractmethod
    async def initialize(self) -> None:
        """初始化向量存储

        创建集合/索引，确保服务可用。
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查

        Returns:
            是否健康
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文档

        Args:
            documents: 文档列表
            ids: 可选的文档 ID 列表

        Returns:
            索引结果
        """
        pass

    @abstractmethod
    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文本

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 可选的文档 ID 列表

        Returns:
            索引结果
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值 (0-1)
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def search_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """通过向量搜索

        Args:
            vector: 查询向量
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> bool:
        """删除文档

        Args:
            ids: 文档 ID 列表

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        """删除整个集合

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    async def get_stats(self) -> VectorStats:
        """获取统计信息

        Returns:
            统计信息
        """
        pass

    async def ensure_initialized(self) -> None:
        """确保已初始化"""
        if not self._initialized:
            await self.initialize()
            self._initialized = True

    def is_initialized(self) -> bool:
        """检查是否已初始化

        Returns:
            是否已初始化
        """
        return self._initialized

    async def as_retriever(
        self,
        k: int = 5,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreRetriever:
        """创建 LangChain 兼容的检索器

        Args:
            k: 返回结果数量
            score_threshold: 相似度阈值
            **kwargs: 额外参数

        Returns:
            VectorStoreRetriever 实例
        """
        from langchain_core.vectorstores import VectorStoreRetriever

        async def _retrieve(query: str) -> list[Document]:
            results = await self.similarity_search(
                query=query,
                k=k,
                score_threshold=score_threshold,
            )
            return [r.to_document() for r in results]

        return VectorStoreRetriever(
            vectorstore=self,  # type: ignore
            search_kwargs={"k": k, "score_threshold": score_threshold},
            _new_retriever=lambda: _retrieve,
        )

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """添加文档（LangChain 兼容接口）

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            文档 ID 列表
        """
        result = await self.add_documents(documents)
        return result.ids


class MemoryVectorStore(BaseVectorStore):
    """内存向量存储

    用于开发/测试环境，使用简单的余弦相似度。
    不持久化，重启后数据丢失。
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化内存向量存储"""
        super().__init__(config, embeddings)
        self._vectors: dict[str, list[float]] = {}
        self._documents: dict[str, Document] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """初始化（内存模式无需额外操作）"""
        self._initialized = True
        logger.info("memory_vector_store_initialized")

    async def health_check(self) -> bool:
        """健康检查"""
        return True

    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> IndexResult:
        """添加文档"""
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        # 生成 ID
        if ids is None:
            import time

            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 嵌入文档
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)

        # 存储
        for doc_id, vector, doc in zip(ids, vectors, documents, strict=True):
            self._vectors[doc_id] = vector
            self._documents[doc_id] = doc
            self._metadata[doc_id] = doc.metadata

        logger.info(
            "memory_store_documents_added",
            count=len(documents),
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

        if not self._vectors:
            return []

        # 嵌入查询
        query_vector = await self.embeddings.aembed_query(query)

        # 计算相似度
        results = self._search_by_vector_internal(
            query_vector, k, score_threshold, filter_dict
        )

        return results

    async def search_by_vector(
        self,
        vector: list[float],
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """通过向量搜索"""
        await self.ensure_initialized()

        if not self._vectors:
            return []

        return self._search_by_vector_internal(vector, k, score_threshold, filter_dict)

    def _search_by_vector_internal(
        self,
        query_vector: list[float],
        k: int,
        score_threshold: float | None,
        filter_dict: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """内部向量搜索"""

        results: list[SearchResult] = []

        for doc_id, vector in self._vectors.items():
            # 应用过滤
            if filter_dict:
                metadata = self._metadata.get(doc_id, {})
                if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                    continue

            # 余弦相似度
            score = self._cosine_similarity(query_vector, vector)

            if score_threshold is None or score >= score_threshold:
                doc = self._documents[doc_id]
                results.append(
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=score,
                        id=doc_id,
                    )
                )

        # 排序并返回 top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        import math

        dot_product = sum(x * y for x, y in zip(a, b, strict=True))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        for doc_id in ids:
            self._vectors.pop(doc_id, None)
            self._documents.pop(doc_id, None)
            self._metadata.pop(doc_id, None)

        logger.info("memory_store_documents_deleted", count=len(ids))
        return True

    async def delete_collection(self) -> bool:
        """删除整个集合"""
        self._vectors.clear()
        self._documents.clear()
        self._metadata.clear()
        logger.info("memory_store_collection_cleared")
        return True

    async def get_stats(self) -> VectorStats:
        """获取统计信息"""
        return VectorStats(
            total_vectors=len(self._vectors),
            collections=1,
            dimension=self.config.dimension,
            metric=self.config.metric,
        )


__all__ = [
    "VectorStoreConfig",
    "SearchResult",
    "IndexResult",
    "VectorStats",
    "BaseVectorStore",
    "MemoryVectorStore",
    "as_langchain_vectorstore",
]


def as_langchain_vectorstore(store: BaseVectorStore) -> LangChainVectorStore:
    """将自定义向量存储转换为 LangChain VectorStore

    Args:
        store: 自定义向量存储实例

    Returns:
        LangChain VectorStore 实例
    """
    return LangChainAdapter(store)


class LangChainAdapter(LangChainVectorStore):
    """LangChain VectorStore 适配器

    将自定义向量存储适配为 LangChain VectorStore。
    """

    def __init__(self, store: BaseVectorStore):
        self._store = store

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        return await self._store.aadd_documents(documents, **kwargs)

    async def aadd_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        ids = [f"doc_{i}" for i in range(len(texts))]
        result = await self._store.add_texts(texts, metadatas, ids)
        return result.ids

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        return self._store.as_retriever(**kwargs)
