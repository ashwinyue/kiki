"""向量存储工厂

根据配置创建向量存储实例。
"""

from typing import Literal

from langchain_core.embeddings import Embeddings

from app.config.settings import get_settings
from app.observability.logging import get_logger
from app.vector_stores.base import BaseVectorStore, MemoryVectorStore, VectorStoreConfig
from app.vector_stores.elasticsearch import ElasticsearchConfig, ElasticsearchVectorStore
from app.vector_stores.pinecone import PineconeConfig, PineconeVectorStore
from app.vector_stores.qdrant import QdrantConfig, QdrantVectorStore

logger = get_logger(__name__)
settings = get_settings()


VectorStoreType = Literal["memory", "qdrant", "pinecone", "elasticsearch"]


class VectorStoreFactory:
    """向量存储工厂

    根据配置创建向量存储实例。
    """

    @staticmethod
    def create(
        store_type: VectorStoreType = "memory",
        config: VectorStoreConfig | None = None,
        embeddings: Embeddings | None = None,
        **kwargs,
    ) -> BaseVectorStore:
        """创建向量存储实例

        Args:
            store_type: 存储类型 (memory, qdrant, pinecone)
            config: 向量存储配置
            embeddings: Embedding 实例
            **kwargs: 额外参数（传递给具体实现）

        Returns:
            向量存储实例

        Examples:
            ```python
            # 内存模式（默认）
            store = VectorStoreFactory.create()

            # Qdrant 本地模式
            store = VectorStoreFactory.create(
                "qdrant",
                config=VectorStoreConfig(collection_name="docs")
            )

            # Pinecone 云端模式
            store = VectorStoreFactory.create(
                "pinecone",
                config=VectorStoreConfig(
                    collection_name="docs",
                    dimension=1536,
                ),
                api_key="xxx",
                environment="gcp-starter",
            )

            # Elasticsearch 模式
            store = VectorStoreFactory.create(
                "elasticsearch",
                config=VectorStoreConfig(collection_name="docs"),
                es_url="http://localhost:9200",
            )
            ```
        """
        config = config or VectorStoreConfig()

        if store_type == "memory":
            return MemoryVectorStore(config, embeddings)

        elif store_type == "qdrant":
            # 构建 Qdrant 配置
            qdrant_config = QdrantConfig(
                collection_name=config.collection_name,
                dimension=config.dimension,
                metric=config.metric,
                embedding_model=config.embedding_model,
                tenant_id=config.tenant_id,
                # Qdrant 特定配置
                url=kwargs.get("url") or getattr(settings, "qdrant_url", None),
                api_key=kwargs.get("api_key") or getattr(settings, "qdrant_api_key", None),
                location=kwargs.get("location", "local"),
                path=kwargs.get("path") or getattr(settings, "qdrant_path", "./data/qdrant"),
                port=kwargs.get("port", 6333),
            )
            return QdrantVectorStore(qdrant_config, embeddings)

        elif store_type == "pinecone":
            # 构建 Pinecone 配置
            pinecone_config = PineconeConfig(
                collection_name=config.collection_name,
                dimension=config.dimension,
                metric=config.metric,
                embedding_model=config.embedding_model,
                tenant_id=config.tenant_id,
                # Pinecone 特定配置
                api_key=kwargs.get("api_key") or getattr(settings, "pinecone_api_key", None),
                environment=kwargs.get("environment"),
                region=kwargs.get("region") or getattr(settings, "pinecone_region", "us-east-1"),
                cloud=kwargs.get("cloud", "aws"),
                namespace=kwargs.get("namespace", ""),
            )
            return PineconeVectorStore(pinecone_config, embeddings)

        elif store_type == "elasticsearch":
            # 构建 Elasticsearch 配置
            es_config = ElasticsearchConfig(
                collection_name=config.collection_name,
                dimension=config.dimension,
                metric=config.metric,
                embedding_model=config.embedding_model,
                tenant_id=config.tenant_id,
                # Elasticsearch 特定配置
                url=kwargs.get("url") or getattr(settings, "elasticsearch_url", None),
                cloud_id=kwargs.get("cloud_id") or getattr(settings, "elasticsearch_cloud_id", None),
                api_key=kwargs.get("api_key") or getattr(settings, "elasticsearch_api_key", None),
                username=kwargs.get("username") or getattr(settings, "elasticsearch_username", None),
                password=kwargs.get("password") or getattr(settings, "elasticsearch_password", None),
                strategy=kwargs.get("strategy", "dense"),
            )
            return ElasticsearchVectorStore(es_config, embeddings)

        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")

    @staticmethod
    def create_from_settings(
        collection_name: str = "default",
        embeddings: Embeddings | None = None,
        tenant_id: int | None = None,
    ) -> BaseVectorStore:
        """从全局配置创建向量存储

        Args:
            collection_name: 集合名称
            embeddings: Embedding 实例
            tenant_id: 租户 ID

        Returns:
            向量存储实例
        """
        # 从配置读取向量存储类型
        store_type: VectorStoreType = getattr(
            settings, "vector_store_type", "memory"
        )  # type: ignore

        # 读取维度配置
        dimension = getattr(settings, "embedding_dimensions", 1024)

        # 创建配置
        config = VectorStoreConfig(
            collection_name=collection_name,
            dimension=dimension,
            tenant_id=tenant_id,
        )

        # 创建向量存储
        return VectorStoreFactory.create(store_type, config, embeddings)


def create_vector_store(
    store_type: VectorStoreType = "memory",
    config: VectorStoreConfig | None = None,
    embeddings: Embeddings | None = None,
    **kwargs,
) -> BaseVectorStore:
    """便捷函数：创建向量存储实例

    Args:
        store_type: 存储类型
        config: 向量存储配置
        embeddings: Embedding 实例
        **kwargs: 额外参数

    Returns:
        向量存储实例
    """
    return VectorStoreFactory.create(store_type, config, embeddings, **kwargs)


__all__ = [
    "VectorStoreType",
    "VectorStoreFactory",
    "create_vector_store",
]
