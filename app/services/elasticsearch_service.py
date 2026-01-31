"""Elasticsearch 服务

提供 Elasticsearch 索引管理、文档 CRUD、搜索等功能。

使用示例:
```python
from app.services.elasticsearch_service import ElasticsearchService

service = ElasticsearchService()

# 创建索引
await service.create_index("documents", dimension=1024)

# 添加文档
await service.add_document("documents", doc_id="1", text="内容", metadata={})

# 搜索
results = await service.search("documents", "查询", k=5)

# 删除索引
await service.delete_index("documents")
```
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.embeddings import Embeddings

from app.config.settings import get_settings
from app.observability.logging import get_logger
from app.vector_stores.base import SearchResult
from app.vector_stores.elasticsearch import (
    ElasticsearchConfig,
    ElasticsearchVectorStore,
)

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class IndexMapping:
    """索引映射配置

    Attributes:
        text_field: 文本字段名
        vector_field: 向量字段名
        metadata_field: 元数据字段名
        dimension: 向量维度
        similarity: 相似度度量
    """

    text_field: str = "text"
    vector_field: str = "vector"
    metadata_field: str = "metadata"
    dimension: int = 1024
    similarity: Literal["cosine", "dot_product", "l2_norm", "max_inner_product"] = "cosine"


@dataclass
class SearchOptions:
    """搜索选项

    Attributes:
        k: 返回结果数量
        score_threshold: 相似度阈值
        filter: 过滤条件
        enable_highlight: 是否启用高亮
        highlight_pre_tag: 高亮前标签
        highlight_post_tag: 高亮后标签
    """

    k: int = 5
    score_threshold: float | None = None
    filter: dict[str, Any] | None = None
    enable_highlight: bool = False
    highlight_pre_tag: str = "<em>"
    highlight_post_tag: str = "</em>"


@dataclass
class IndexStats:
    """索引统计信息

    Attributes:
        index_name: 索引名称
        doc_count: 文档数量
        store_size: 存储大小（字节）
        dimension: 向量维度
        health: 健康状态
        status: 状态
    """

    index_name: str
    doc_count: int
    store_size: int
    dimension: int
    health: str
    status: str


@dataclass
class BulkResult:
    """批量操作结果

    Attributes:
        total: 总数
        successful: 成功数量
        failed: 失败数量
        errors: 错误列表
    """

    total: int
    successful: int
    failed: int
    errors: list[dict[str, Any]] = field(default_factory=list)


class ElasticsearchService:
    """Elasticsearch 服务

    提供完整的 Elasticsearch 功能封装。
    """

    def __init__(
        self,
        config: ElasticsearchConfig | None = None,
        embeddings: Embeddings | None = None,
        tenant_id: int | None = None,
    ):
        """初始化服务

        Args:
            config: Elasticsearch 配置
            embeddings: Embedding 实例
            tenant_id: 租户 ID
        """
        self.embeddings = embeddings
        self.tenant_id = tenant_id

        # 构建配置
        if config:
            self.config = config
        else:
            self.config = self._build_config_from_settings()

        # 创建向量存储实例
        self.store = ElasticsearchVectorStore(
            config=self.config,
            embeddings=embeddings,
        )

    def _build_config_from_settings(self) -> ElasticsearchConfig:
        """从设置构建配置

        Returns:
            Elasticsearch 配置
        """
        return ElasticsearchConfig(
            url=getattr(settings, "elasticsearch_url", "http://localhost:9200"),
            cloud_id=getattr(settings, "elasticsearch_cloud_id", None),
            api_key=getattr(settings, "elasticsearch_api_key", None),
            username=getattr(settings, "elasticsearch_username", None),
            password=getattr(settings, "elasticsearch_password", None),
            tenant_id=self.tenant_id,
            dimension=getattr(settings, "embedding_dimensions", 1024),
            strategy=getattr(settings, "elasticsearch_strategy", "dense"),
        )

    # ========== 初始化 ==========

    async def initialize(self) -> None:
        """初始化服务"""
        await self.store.initialize()
        logger.info("elasticsearch_service_initialized")

    async def health_check(self) -> dict[str, Any]:
        """健康检查

        Returns:
            健康状态信息
        """
        is_healthy = await self.store.health_check()

        info: dict[str, Any] = {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": time.time(),
        }

        if is_healthy:
            # 获取集群信息
            if self.store._client:
                try:
                    cluster_info = await self.store._client.info()
                    info["cluster_name"] = cluster_info.get("cluster_name")
                    info["version"] = cluster_info.get("version", {}).get("number")
                except Exception as e:
                    logger.warning("elasticsearch_cluster_info_failed", error=str(e))

        return info

    # ========== 索引管理 ==========

    async def create_index(
        self,
        index_name: str,
        mapping: IndexMapping | None = None,
        force: bool = False,
    ) -> bool:
        """创建索引

        Args:
            index_name: 索引名称
            mapping: 索引映射
            force: 是否强制重建（删除已存在的索引）

        Returns:
            是否成功
        """
        await self.store.ensure_initialized()

        client = self.store._client

        # 检查索引是否存在
        exists = await client.indices.exists(index=index_name)

        if exists:
            if force:
                await client.indices.delete(index=index_name)
                logger.info("elasticsearch_index_deleted_for_recreate", index=index_name)
            else:
                logger.warning("elasticsearch_index_already_exists", index=index_name)
                return True

        # 构建映射
        if mapping is None:
            mapping = IndexMapping(dimension=self.config.dimension)

        mapping_body = self._build_mapping(mapping)

        # 创建索引
        await client.indices.create(
            index=index_name,
            body={"mappings": mapping_body},
        )

        logger.info("elasticsearch_index_created", index=index_name)
        return True

    def _build_mapping(self, mapping: IndexMapping) -> dict[str, Any]:
        """构建索引映射

        Args:
            mapping: 映射配置

        Returns:
            Elasticsearch 映射体
        """
        properties: dict[str, Any] = {
            mapping.text_field: {"type": "text"},
            mapping.metadata_field: {"type": "object", "dynamic": True},
        }

        # 添加租户字段
        if self.tenant_id:
            properties["tenant_id"] = {"type": "keyword"}

        # 添加向量字段
        if self.config.strategy in ("dense", "hybrid"):
            properties[mapping.vector_field] = {
                "type": "dense_vector",
                "dims": mapping.dimension,
                "index": True,
                "similarity": mapping.similarity,
            }

        return {"properties": properties}

    async def delete_index(self, index_name: str) -> bool:
        """删除索引

        Args:
            index_name: 索引名称

        Returns:
            是否成功
        """
        await self.store.ensure_initialized()

        client = self.store._client
        exists = await client.indices.exists(index=index_name)

        if not exists:
            logger.warning("elasticsearch_index_not_found", index=index_name)
            return False

        await client.indices.delete(index=index_name)
        logger.info("elasticsearch_index_deleted", index=index_name)
        return True

    async def list_indices(self) -> list[str]:
        """列出所有索引

        Returns:
            索引名称列表
        """
        await self.store.ensure_initialized()

        client = self.store._client
        response = await client.indices.get(index="*")

        return list(response.keys())

    async def get_index_stats(self, index_name: str) -> IndexStats | None:
        """获取索引统计

        Args:
            index_name: 索引名称

        Returns:
            索引统计信息
        """
        await self.store.ensure_initialized()

        client = self.store._client

        # 检查索引是否存在
        exists = await client.indices.exists(index=index_name)
        if not exists:
            return None

        # 获取统计
        stats = await client.indices.stats(index=index_name)
        index_stats = stats["indices"][index_name]

        # 获取设置（用于验证索引存在）
        await client.indices.get_settings(index=index_name)

        return IndexStats(
            index_name=index_name,
            doc_count=index_stats["primaries"]["docs"]["count"],
            store_size=index_stats["primaries"]["store"]["size_in_bytes"],
            dimension=self.config.dimension,
            health="green",
            status="open",
        )

    async def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在

        Args:
            index_name: 索引名称

        Returns:
            是否存在
        """
        await self.store.ensure_initialized()

        client = self.store._client
        return await client.indices.exists(index=index_name)

    # ========== 文档操作 ==========

    async def add_document(
        self,
        index_name: str,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """添加单个文档

        Args:
            index_name: 索引名称
            doc_id: 文档 ID
            text: 文本内容
            metadata: 元数据

        Returns:
            是否成功
        """
        result = await self.add_documents(
            index_name=index_name,
            documents=[(text, metadata)],
            ids=[doc_id],
        )
        return result.failed == 0

    async def add_documents(
        self,
        index_name: str,
        documents: list[tuple[str, dict[str, Any] | None]],
        ids: list[str] | None = None,
    ) -> BulkResult:
        """批量添加文档

        Args:
            index_name: 索引名称
            documents: (text, metadata) 元组列表
            ids: 文档 ID 列表

        Returns:
            批量操作结果
        """
        await self.store.ensure_initialized()

        client = self.store._client

        # 生成 ID
        if ids is None:
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 构建批量操作
        actions = []
        for doc_id, (text, metadata) in zip(ids, documents, strict=True):
            doc_metadata = metadata or {}
            if self.tenant_id:
                doc_metadata["tenant_id"] = self.tenant_id

            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc_id,
                    "text": text,
                    "metadata": doc_metadata,
                }
            )

        # 执行批量操作
        from elasticsearch.helpers import async_bulk

        success = 0
        failed = 0
        errors = []

        async for ok, response in async_bulk(
            client,
            actions,
            raise_on_error=False,
        ):
            if ok:
                success += 1
            else:
                failed += 1
                errors.append(response)

        logger.info(
            "elasticsearch_documents_indexed",
            index=index_name,
            success=success,
            failed=failed,
        )

        return BulkResult(
            total=len(documents),
            successful=success,
            failed=failed,
            errors=errors,
        )

    async def get_document(
        self,
        index_name: str,
        doc_id: str,
    ) -> dict[str, Any] | None:
        """获取文档

        Args:
            index_name: 索引名称
            doc_id: 文档 ID

        Returns:
            文档内容
        """
        await self.store.ensure_initialized()

        client = self.store._client

        try:
            response = await client.get(index=index_name, id=doc_id)
            return response["_source"]
        except Exception:
            return None

    async def update_document(
        self,
        index_name: str,
        doc_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新文档

        Args:
            index_name: 索引名称
            doc_id: 文档 ID
            text: 新文本内容（可选）
            metadata: 新元数据（可选）

        Returns:
            是否成功
        """
        await self.store.ensure_initialized()

        client = self.store._client

        doc: dict[str, Any] = {"doc": {}}

        if text is not None:
            doc["doc"]["text"] = text

        if metadata is not None:
            doc["doc"]["metadata"] = metadata

        if not doc["doc"]:
            return False

        try:
            await client.update(index=index_name, id=doc_id, body=doc)
            logger.info("elasticsearch_document_updated", index=index_name, doc_id=doc_id)
            return True
        except Exception:
            return False

    async def delete_document(
        self,
        index_name: str,
        doc_id: str,
    ) -> bool:
        """删除文档

        Args:
            index_name: 索引名称
            doc_id: 文档 ID

        Returns:
            是否成功
        """
        await self.store.ensure_initialized()

        client = self.store._client

        try:
            await client.delete(index=index_name, id=doc_id)
            logger.info("elasticsearch_document_deleted", index=index_name, doc_id=doc_id)
            return True
        except Exception:
            return False

    async def delete_by_query(
        self,
        index_name: str,
        query: dict[str, Any],
    ) -> int:
        """按查询删除文档

        Args:
            index_name: 索引名称
            query: 删除查询

        Returns:
            删除的文档数量
        """
        await self.store.ensure_initialized()

        client = self.store._client

        # 添加租户过滤
        if self.tenant_id:
            query = {
                "bool": {
                    "must": query,
                    "filter": [{"term": {"tenant_id": self.tenant_id}}],
                }
            }

        response = await client.delete_by_query(
            index=index_name,
            body={"query": query},
        )

        deleted = response.get("deleted", 0)
        logger.info("elasticsearch_documents_deleted_by_query", index=index_name, count=deleted)
        return deleted

    # ========== 搜索功能 ==========

    async def search(
        self,
        index_name: str,
        query: str,
        options: SearchOptions | None = None,
    ) -> list[SearchResult]:
        """搜索文档

        Args:
            index_name: 索引名称
            query: 查询文本
            options: 搜索选项

        Returns:
            搜索结果列表
        """
        await self.store.ensure_initialized()

        # 更新索引名称
        self.config.index_name = index_name
        self.store.es_config.index_name = index_name

        options = options or SearchOptions()

        # 使用向量存储搜索
        return await self.store.similarity_search(
            query=query,
            k=options.k,
            score_threshold=options.score_threshold,
            filter_dict=options.filter,
        )

    async def hybrid_search(
        self,
        index_name: str,
        query: str,
        k: int = 5,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """混合搜索

        Args:
            index_name: 索引名称
            query: 查询文本
            k: 返回结果数量
            text_weight: 文本搜索权重
            vector_weight: 向量搜索权重
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        await self.store.ensure_initialized()

        self.config.index_name = index_name
        self.store.es_config.index_name = index_name

        return await self.store.hybrid_search(
            query=query,
            k=k,
            text_weight=text_weight,
            vector_weight=vector_weight,
            filter_dict=filter_dict,
        )

    async def raw_search(
        self,
        index_name: str,
        query: dict[str, Any],
    ) -> dict[str, Any]:
        """执行原始查询

        Args:
            index_name: 索引名称
            query: Elasticsearch 查询 DSL

        Returns:
            搜索结果
        """
        await self.store.ensure_initialized()

        client = self.store._client

        response = await client.search(
            index=index_name,
            body=query,
        )

        return response

    # ========== 分析器 ==========

    async def analyze(
        self,
        index_name: str,
        text: str,
        analyzer: str | None = None,
    ) -> list[dict[str, Any]]:
        """分析文本

        Args:
            index_name: 索引名称
            text: 待分析文本
            analyzer: 分析器名称

        Returns:
            分词结果
        """
        await self.store.ensure_initialized()

        client = self.store._client

        body = {"text": text}
        if analyzer:
            body["analyzer"] = analyzer

        try:
            response = await client.indices.analyze(
                index=index_name,
                body=body,
            )
            return response.get("tokens", [])
        except Exception:
            return []

    # ========== 关闭 ==========

    async def close(self) -> None:
        """关闭服务"""
        await self.store.close()
        logger.info("elasticsearch_service_closed")


__all__ = [
    "ElasticsearchService",
    "IndexMapping",
    "SearchOptions",
    "IndexStats",
    "BulkResult",
]
