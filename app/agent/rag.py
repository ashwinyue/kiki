"""RAG 检索模块

提供向量存储和检索增强生成功能。
支持多种向量数据库：PostgreSQL (pgvector)、Pinecone、Chroma。

依赖安装:
    # PostgreSQL + pgvector
    uv add -E pgvector langchain-postgres pgvector

    # Pinecone
    uv add -E pinecone langchain-pinecone pinecone

    # Chroma
    uv add -E chroma langchain-chroma chromadb

使用示例:
```python
from app.agent.rag import (
    VectorStore,
    retrieve_documents,
    create_retrieval_tool,
)

# 初始化向量存储
vector_store = VectorStore.from_documents(
    documents=texts,
    collection_name="knowledge_base",
)

# 检索文档
results = await vector_store.asimilarity_search("查询内容", k=5)

# 创建检索工具
retrieval_tool = create_retrieval_tool(vector_store)
```
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as LangChainVectorStore

from app.config.settings import get_settings
from app.llm.embeddings import get_embeddings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ============== 向量存储配置 ==============

@dataclass
class VectorStoreConfig:
    """向量存储配置

    Attributes:
        collection_name: 集合名称
        embedding_model: Embedding 模型名称
        dimension: 向量维度
        metric: 相似度度量方式
    """

    collection_name: str = "default"
    embedding_model: str = "text-embedding-v4"
    dimension: int = 1024
    metric: str = "cosine"  # cosine, l2, max_inner_product


# ============== 检索结果 ==============

@dataclass
class SearchResult:
    """检索结果

    Attributes:
        content: 文档内容
        metadata: 元数据
        score: 相似度得分
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


# ============== 向量存储抽象 ==============

class BaseVectorStore(ABC):
    """向量存储抽象基类

    定义向量存储的通用接口。
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
        self.embeddings = embeddings or get_embeddings(
            provider="dashscope",
            model=self.config.embedding_model,
            dimensions=self.config.dimension,
        )
        self._store: LangChainVectorStore | None = None

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档

        Args:
            documents: 文档列表

        Returns:
            文档 ID 列表
        """
        pass

    @abstractmethod
    async def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本

        Args:
            texts: 文本列表
            metadatas: 元数据列表

        Returns:
            文档 ID 列表
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """异步相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值

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

    async def aensure_index(self) -> None:
        """确保索引已创建"""
        if self._store is None:
            await self._initialize_store()

    async def _initialize_store(self) -> None:
        """初始化底层向量存储（子类实现）"""
        pass

    def is_initialized(self) -> bool:
        """检查是否已初始化

        Returns:
            是否已初始化
        """
        return self._store is not None


# ============== PostgreSQL + pgvector 实现 ==============

class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector 向量存储

    使用 PostgreSQL 的 pgvector 扩展进行向量检索。
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
        connection_string: str | None = None,
    ):
        """初始化 PgVector 向量存储

        Args:
            config: 向量存储配置
            embeddings: Embedding 实例
            connection_string: PostgreSQL 连接字符串
        """
        super().__init__(config, embeddings)

        self.connection_string = connection_string or (
            f"postgresql://{settings.database_user}:{settings.database_password}"
            f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
        )

    async def _initialize_store(self) -> None:
        """初始化底层向量存储"""
        try:
            from langchain_postgres import PGVector

            self._store = await PGVector.ainit(
                collection_name=self.config.collection_name,
                connection=self.connection_string,
                embeddings=self.embeddings,
                dimension=self.config.dimension,
            )

            logger.info(
                "pgvector_initialized",
                collection=self.config.collection_name,
            )

        except ImportError:
            logger.error("langchain_postgres_not_installed")
            raise ImportError(
                "请安装 langchain-postgres: uv add -E pgvector langchain-postgres pgvector"
            )
        except Exception as e:
            logger.error("pgvector_init_failed", error=str(e))
            raise

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档"""
        await self.aensure_index()

        ids = [
            f"{self.config.collection_name}_{datetime.now().timestamp()}_{i}"
            for i in range(len(documents))
        ]

        await self._store.aadd_documents(documents, ids=ids)
        return ids

    async def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        documents = [
            Document(page_content=text, metadata=(metadatas or [{}])[i])
            for i, text in enumerate(texts)
        ]
        return await self.add_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """相似度搜索"""
        await self.aensure_index()

        results = await self._store.asimilarity_search_with_score(query, k=k)

        search_results = []
        for doc, score in results:
            if score_threshold is None or score >= score_threshold:
                search_results.append(
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=score,
                    )
                )

        return search_results

    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """异步相似度搜索"""
        return await self.similarity_search(query, k, score_threshold)

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.aensure_index()

        try:
            await self._store.adelete(ids)
            return True
        except Exception as e:
            logger.error("pgvector_delete_failed", error=str(e))
            return False


# ============== Pinecone 实现 ==============

class PineconeStore(BaseVectorStore):
    """Pinecone 向量存储

    使用 Pinecone 云服务进行向量检索。
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
        api_key: str | None = None,
        environment: str | None = None,
    ):
        """初始化 Pinecone 向量存储

        Args:
            config: 向量存储配置
            embeddings: Embedding 实例
            api_key: Pinecone API Key
            environment: Pinecone 环境
        """
        super().__init__(config, embeddings)
        self.api_key = api_key
        self.environment = environment

    async def _initialize_store(self) -> None:
        """初始化底层向量存储"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore

            # 初始化 Pinecone
            pc = Pinecone(api_key=self.api_key)

            # 检查索引是否存在
            if self.config.collection_name not in [idx.name for idx in pc.list_indexes()]:
                pc.create_index(
                    name=self.config.collection_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        environment=self.environment or "gcp-starter",
                        cloud="aws",
                    ),
                )

            self._store = PineconeVectorStore(
                index_name=self.config.collection_name,
                embedding=self.embeddings,
            )

            logger.info(
                "pinecone_initialized",
                collection=self.config.collection_name,
            )

        except ImportError:
            logger.error("langchain_pinecone_not_installed")
            raise ImportError(
                "请安装 langchain-pinecone: uv add -E pinecone langchain-pinecone pinecone"
            )
        except Exception as e:
            logger.error("pinecone_init_failed", error=str(e))
            raise

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档"""
        await self.aensure_index()

        ids = [
            f"{self.config.collection_name}_{datetime.now().timestamp()}_{i}"
            for i in range(len(documents))
        ]

        await self._store.aadd_documents(documents, ids=ids)
        return ids

    async def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        documents = [
            Document(page_content=text, metadata=(metadatas or [{}])[i])
            for i, text in enumerate(texts)
        ]
        return await self.add_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """相似度搜索"""
        await self.aensure_index()

        results = await self._store.asimilarity_search_with_score(query, k=k)

        search_results = []
        for doc, score in results:
            if score_threshold is None or score >= score_threshold:
                search_results.append(
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=score,
                    )
                )

        return search_results

    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """异步相似度搜索"""
        return await self.similarity_search(query, k, score_threshold)

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.aensure_index()

        try:
            await self._store.adelete(ids)
            return True
        except Exception as e:
            logger.error("pinecone_delete_failed", error=str(e))
            return False


# ============== Chroma 实现 ==============

class ChromaStore(BaseVectorStore):
    """Chroma 向量存储

    使用本地 Chroma 数据库进行向量检索。
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
        persist_directory: str | None = None,
    ):
        """初始化 Chroma 向量存储

        Args:
            config: 向量存储配置
            embeddings: Embedding 实例
            persist_directory: 持久化目录
        """
        super().__init__(config, embeddings)
        self.persist_directory = persist_directory or "./chroma_db"

    async def _initialize_store(self) -> None:
        """初始化底层向量存储"""
        try:
            from langchain_chroma import Chroma
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)

            self._store = Chroma(
                client=client,
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
            )

            logger.info(
                "chroma_initialized",
                collection=self.config.collection_name,
                persist_directory=self.persist_directory,
            )

        except ImportError:
            logger.error("langchain_chroma_not_installed")
            raise ImportError(
                "请安装 langchain-chroma: uv add -E chroma langchain-chroma chromadb"
            )
        except Exception as e:
            logger.error("chroma_init_failed", error=str(e))
            raise

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档"""
        await self.aensure_index()

        ids = [
            f"{self.config.collection_name}_{datetime.now().timestamp()}_{i}"
            for i in range(len(documents))
        ]

        await self._store.aadd_documents(documents, ids=ids)
        return ids

    async def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """添加文本"""
        documents = [
            Document(page_content=text, metadata=(metadatas or [{}])[i])
            for i, text in enumerate(texts)
        ]
        return await self.add_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """相似度搜索"""
        await self.aensure_index()

        results = await self._store.asimilarity_search_with_score(query, k=k)

        search_results = []
        for doc, score in results:
            if score_threshold is None or score >= score_threshold:
                search_results.append(
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=score,
                    )
                )

        return search_results

    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """异步相似度搜索"""
        return await self.similarity_search(query, k, score_threshold)

    async def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        await self.aensure_index()

        try:
            await self._store.adelete(ids)
            return True
        except Exception as e:
            logger.error("chroma_delete_failed", error=str(e))
            return False


# ============== 向量存储工厂 ==============

class VectorStoreType:
    """向量存储类型"""

    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    CHROMA = "chroma"


def create_vector_store(
    store_type: str,
    config: VectorStoreConfig | None = None,
    **kwargs,
) -> BaseVectorStore:
    """创建向量存储实例

    Args:
        store_type: 存储类型 (pgvector/pinecone/chroma)
        config: 向量存储配置
        **kwargs: 额外参数

    Returns:
        向量存储实例

    Examples:
        ```python
        # PostgreSQL + pgvector
        store = create_vector_store("pgvector")

        # Pinecone
        store = create_vector_store(
            "pinecone",
            api_key="xxx",
            environment="gcp-starter"
        )

        # Chroma (本地)
        store = create_vector_store("chroma", persist_directory="./db")
        ```
    """
    if store_type == VectorStoreType.PGVECTOR:
        return PgVectorStore(config, **kwargs)
    elif store_type == VectorStoreType.PINECONE:
        return PineconeStore(config, **kwargs)
    elif store_type == VectorStoreType.CHROMA:
        return ChromaStore(config, **kwargs)
    else:
        raise ValueError(f"不支持的向量存储类型: {store_type}")


# ============== 便捷函数 ==============

async def retrieve_documents(
    query: str,
    store: BaseVectorStore | None = None,
    k: int = 5,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """检索文档

    Args:
        query: 查询文本
        store: 向量存储实例（默认使用 Chroma）
        k: 返回结果数量
        score_threshold: 相似度阈值

    Returns:
        搜索结果列表

    Examples:
        ```python
        results = await retrieve_documents("Python 编程", k=3)
        for result in results:
            print(f"{result.score}: {result.content[:100]}...")
        ```
    """
    if store is None:
        # 默认使用 Chroma
        store = ChromaStore()

    return await store.asimilarity_search(query, k, score_threshold)


async def index_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
    store: BaseVectorStore | None = None,
    collection_name: str = "default",
) -> list[str]:
    """索引文档

    Args:
        texts: 文本列表
        metadatas: 元数据列表
        store: 向量存储实例
        collection_name: 集合名称

    Returns:
        文档 ID 列表

    Examples:
        ```python
        texts = ["文档1内容", "文档2内容", "文档3内容"]
        metadatas = [
            {"source": "doc1.pdf", "page": 1},
            {"source": "doc2.pdf", "page": 1},
            {"source": "doc3.pdf", "page": 1},
        ]
        ids = await index_documents(texts, metadatas)
        ```
    """
    config = VectorStoreConfig(collection_name=collection_name)

    if store is None:
        store = ChromaStore(config)

    return await store.add_texts(texts, metadatas)


__all__ = [
    # 配置
    "VectorStoreConfig",
    # 检索结果
    "SearchResult",
    # 向量存储
    "BaseVectorStore",
    "PgVectorStore",
    "PineconeStore",
    "ChromaStore",
    "VectorStoreType",
    # 工厂函数
    "create_vector_store",
    # 便捷函数
    "retrieve_documents",
    "index_documents",
]
