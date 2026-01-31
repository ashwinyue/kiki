"""Elasticsearch 向量存储

支持 Elasticsearch 8.x 向量搜索功能。
文档: https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/

依赖安装:
    uv add langchain-elasticsearch "elasticsearch[async]>=8.0.0"

特性:
- DenseVectorStrategy: 密集向量搜索（语义搜索）
- SparseVectorStrategy: 稀疏向量搜索（ELSER）
- 混合搜索: 结合密集向量和 BM25
"""

from dataclasses import dataclass
from typing import Any, Literal

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
class ElasticsearchConfig(VectorStoreConfig):
    """Elasticsearch 配置

    Attributes:
        url: Elasticsearch 服务器 URL
        cloud_id: Elastic Cloud ID（云端部署）
        api_key: API Key（替代用户名/密码）
        username: 用户名（基础认证）
        password: 密码（基础认证）
        index_name: 索引名称（覆盖 collection_name）
        strategy: 向量策略类型
        similarity: 相似度度量
        hybrid_search: 是否启用混合搜索
        rerank: 是否启用重排序
        verify_certs: 是否验证证书
        request_timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
    """

    url: str | None = None
    cloud_id: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    index_name: str | None = None

    # 向量策略
    strategy: Literal["dense", "sparse", "hybrid"] = "dense"
    similarity: Literal["cosine", "dot_product", "l2_norm", "max_inner_product"] = "cosine"

    # 混合搜索
    hybrid_search: bool = False
    rerank: bool = False

    # 连接配置
    verify_certs: bool = True
    request_timeout: int = 30
    max_retries: int = 3


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch 向量存储

    支持密集向量、稀疏向量和混合搜索。
    """

    def __init__(
        self,
        config: ElasticsearchConfig | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化 Elasticsearch 向量存储

        Args:
            config: Elasticsearch 配置
            embeddings: Embedding 实例
        """
        super().__init__(config, embeddings)
        self.es_config: ElasticsearchConfig = config or ElasticsearchConfig()
        self._client: Any = None
        self._store: Any = None

    async def initialize(self) -> None:
        """初始化 Elasticsearch 客户端和索引"""
        try:
            from langchain_elasticsearch import DenseVectorStrategy
            from langchain_elasticsearch.elasticsearch import ElasticsearchStore

            # 构建连接参数
            es_client_params = self._build_client_params()

            # 创建异步客户端
            from elasticsearch import AsyncElasticsearch

            self._client = AsyncElasticsearch(**es_client_params)

            # 测试连接
            await self._client.ping()
            logger.info("elasticsearch_client_connected", url=self.es_config.url)

            # 确定索引名称
            index_name = self.es_config.index_name or self.config.collection_name

            # 选择向量策略
            strategy = self._get_strategy()

            # 创建 LangChain ElasticsearchStore
            self._store = ElasticsearchStore(
                es_connection=self._client,
                index_name=index_name,
                strategy=strategy,
                embedding=self.embeddings if strategy == DenseVectorStrategy() else None,
            )

            # 确保索引存在
            await self._ensure_index()

            self._initialized = True
            logger.info(
                "elasticsearch_initialized",
                index=index_name,
                strategy=self.es_config.strategy,
            )

        except ImportError as e:
            logger.error("elasticsearch_not_installed")
            raise ImportError(
                "请安装依赖: uv add langchain-elasticsearch 'elasticsearch[async]>=8.0.0'"
            ) from e
        except Exception as e:
            logger.error("elasticsearch_init_failed", error=str(e))
            raise

    def _build_client_params(self) -> dict[str, Any]:
        """构建 Elasticsearch 客户端参数

        Returns:
            客户端参数字典
        """
        params: dict[str, Any] = {
            "verify_certs": self.es_config.verify_certs,
            "request_timeout": self.es_config.request_timeout,
            "max_retries": self.es_config.max_retries,
        }

        if self.es_config.cloud_id:
            params["cloud_id"] = self.es_config.cloud_id
        elif self.es_config.url:
            params["hosts"] = [self.es_config.url]

        # 认证
        if self.es_config.api_key:
            params["api_key"] = self.es_config.api_key
        elif self.es_config.username and self.es_config.password:
            params["basic_auth"] = (self.es_config.username, self.es_config.password)

        return params

    def _get_strategy(self) -> Any:
        """获取向量策略

        Returns:
            策略实例
        """
        from langchain_elasticsearch import (
            DenseVectorStrategy,
            SparseVectorStrategy,
        )

        if self.es_config.strategy == "dense":
            return DenseVectorStrategy(
                similarity=self._map_similarity(),
            )
        elif self.es_config.strategy == "sparse":
            return SparseVectorStrategy()
        elif self.es_config.strategy == "hybrid":
            # 混合策略结合密集向量和稀疏向量
            return DenseVectorStrategy(
                similarity=self._map_similarity(),
                hybrid=True,
            )
        else:
            return DenseVectorStrategy()

    def _map_similarity(self) -> str:
        """映射相似度度量

        Returns:
            Elasticsearch 相似度参数
        """
        mapping = {
            "cosine": "cosine",
            "dot_product": "dot_product",
            "l2_norm": "l2_norm",
            "max_inner_product": "max_inner_product",
        }
        return mapping.get(self.es_config.similarity, "cosine")

    async def _ensure_index(self) -> None:
        """确保索引存在

        如果索引不存在则创建。
        """
        index_name = self.es_config.index_name or self.config.collection_name

        exists = await self._client.indices.exists(index=index_name)

        if not exists:
            # 创建索引映射
            mapping = self._build_index_mapping()
            await self._client.indices.create(
                index=index_name,
                body={"mappings": mapping},
            )
            logger.info("elasticsearch_index_created", index=index_name)

    def _build_index_mapping(self) -> dict[str, Any]:
        """构建索引映射

        Returns:
            映射配置
        """
        mapping: dict[str, Any] = {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"type": "object", "dynamic": True},
            }
        }

        # 添加租户字段
        if self.config.tenant_id:
            mapping["properties"]["tenant_id"] = {"type": "keyword"}

        # 根据策略添加向量字段
        if self.es_config.strategy in ("dense", "hybrid"):
            mapping["properties"]["vector"] = {
                "type": "dense_vector",
                "dims": self.config.dimension,
                "index": True,
                "similarity": self._map_similarity(),
            }

        if self.es_config.strategy in ("sparse", "hybrid"):
            mapping["properties"]["sparse_vector"] = {
                "type": "sparse_vector",
            }

        return mapping

    async def health_check(self) -> bool:
        """健康检查

        Returns:
            是否健康
        """
        try:
            if self._client is None:
                return False
            return await self._client.ping()
        except Exception as e:
            logger.warning("elasticsearch_health_check_failed", error=str(e))
            return False

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
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        import time

        # 生成 ID
        if ids is None:
            timestamp = int(time.time() * 1000)
            ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]

        # 添加租户 ID 到元数据
        if self.config.tenant_id:
            for doc in documents:
                doc.metadata["tenant_id"] = self.config.tenant_id

        # 使用 LangChain store 添加文档
        try:

            # 准备批量数据
            bulk_data = []
            for doc_id, doc in zip(ids, documents, strict=True):
                bulk_data.append(
                    {
                        "_op_type": "index",
                        "_id": doc_id,
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )

            # 批量索引
            index_name = self.es_config.index_name or self.config.collection_name
            success, failed = await self._bulk_index(bulk_data, index_name)

            logger.info(
                "elasticsearch_documents_added",
                count=success,
                failed=failed,
                index=index_name,
            )

            return IndexResult(
                ids=ids[:success],
                count=success,
                failed=failed,
            )

        except Exception as e:
            logger.error("elasticsearch_add_documents_failed", error=str(e))
            raise

    async def _bulk_index(
        self,
        actions: list[dict[str, Any]],
        index_name: str,
    ) -> tuple[int, int]:
        """批量索引文档

        Args:
            actions: 操作列表
            index_name: 索引名称

        Returns:
            (成功数量, 失败数量)
        """
        from elasticsearch.helpers import async_bulk

        success = 0
        failed = 0

        async for ok, response in async_bulk(
            self._client,
            actions,
            index=index_name,
            raise_on_error=False,
        ):
            if ok:
                success += 1
            else:
                failed += 1
                logger.warning("bulk_index_failed", response=response)

        return success, failed

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
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值 (0-1)
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        try:
            # 构建过滤查询
            filter_query = self._build_filter_query(filter_dict)

            # 使用 LangChain store 搜索
            docs_with_scores = await self._store.asimilarity_search_with_score(
                query,
                k=k,
                filter=filter_query if filter_query else None,
            )

            # 转换结果
            results = []
            for doc, score in docs_with_scores:
                # 应用阈值过滤
                if score_threshold is not None and score < score_threshold:
                    continue

                results.append(
                    SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=score,
                    )
                )

            return results

        except Exception as e:
            logger.error("elasticsearch_similarity_search_failed", error=str(e))
            return []

    def _build_filter_query(self, filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
        """构建过滤查询

        Args:
            filter_dict: 过滤条件

        Returns:
            Elasticsearch 查询 DSL
        """
        if not filter_dict and not self.config.tenant_id:
            return None

        must = []

        # 租户过滤
        if self.config.tenant_id:
            must.append({"term": {"tenant_id": self.config.tenant_id}})

        # 额外过滤
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, (list, tuple)):
                    must.append({"terms": {f"metadata.{key}": list(value)}})
                elif isinstance(value, dict):
                    # 支持范围查询等
                    must.append({**value, "key": f"metadata.{key}"})
                else:
                    must.append({"term": {f"metadata.{key}": value}})

        return {"bool": {"must": must}} if must else None

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
        await self.ensure_initialized()

        try:
            index_name = self.es_config.index_name or self.config.collection_name
            filter_query = self._build_filter_query(filter_dict)

            # 构建查询
            query: dict[str, Any] = {
                "knn": {
                    "field": "vector",
                    "query_vector": vector,
                    "k": k,
                    "num_candidates": k * 10,
                }
            }

            # 添加过滤
            if filter_query:
                query["query"] = filter_query

            # 搜索
            response = await self._client.search(
                index=index_name,
                body=query,
                size=k,
            )

            # 转换结果
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                source = hit["_source"]

                # 应用阈值
                if score_threshold is not None and score < score_threshold:
                    continue

                results.append(
                    SearchResult(
                        content=source.get("text", ""),
                        metadata=source.get("metadata", {}),
                        score=score,
                        id=hit.get("_id"),
                    )
                )

            return results

        except Exception as e:
            logger.error("elasticsearch_search_by_vector_failed", error=str(e))
            return []

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """混合搜索

        结合 BM25 关键词搜索和向量语义搜索。

        Args:
            query: 查询文本
            k: 返回结果数量
            text_weight: 文本搜索权重
            vector_weight: 向量搜索权重
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        await self.ensure_initialized()

        if self.embeddings is None:
            raise ValueError("Embeddings not configured")

        try:
            index_name = self.es_config.index_name or self.config.collection_name
            filter_query = self._build_filter_query(filter_dict)

            # 嵌入查询
            query_vector = await self.embeddings.aembed_query(query)

            # 构建混合查询
            hybrid_query: dict[str, Any] = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            # BM25 查询
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": text_weight,
                                    }
                                }
                            },
                            # KNN 查询
                            {
                                "knn": {
                                    "field": "vector",
                                    "query_vector": query_vector,
                                    "k": k,
                                    "boost": vector_weight,
                                }
                            },
                        ]
                    }
                },
            }

            # 添加过滤
            if filter_query:
                hybrid_query["query"]["bool"]["filter"] = filter_query.get("bool", {}).get("must", [])

            # 搜索
            response = await self._client.search(
                index=index_name,
                body=hybrid_query,
            )

            # 转换结果
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    SearchResult(
                        content=source.get("text", ""),
                        metadata=source.get("metadata", {}),
                        score=hit.get("_score", 0.0),
                        id=hit.get("_id"),
                    )
                )

            return results

        except Exception as e:
            logger.error("elasticsearch_hybrid_search_failed", error=str(e))
            return []

    async def delete(self, ids: list[str]) -> bool:
        """删除文档

        Args:
            ids: 文档 ID 列表

        Returns:
            是否成功
        """
        await self.ensure_initialized()

        try:
            index_name = self.es_config.index_name or self.config.collection_name

            # 批量删除
            await self._client.delete_by_query(
                index=index_name,
                body={"query": {"ids": {"values": ids}}},
            )

            logger.info(
                "elasticsearch_documents_deleted",
                count=len(ids),
                index=index_name,
            )
            return True

        except Exception as e:
            logger.error("elasticsearch_delete_failed", error=str(e))
            return False

    async def delete_collection(self) -> bool:
        """删除整个集合

        Returns:
            是否成功
        """
        await self.ensure_initialized()

        try:
            index_name = self.es_config.index_name or self.config.collection_name

            exists = await self._client.indices.exists(index=index_name)
            if exists:
                await self._client.indices.delete(index=index_name)
                logger.info("elasticsearch_index_deleted", index=index_name)

            return True

        except Exception as e:
            logger.error("elasticsearch_delete_collection_failed", error=str(e))
            return False

    async def get_stats(self) -> VectorStats:
        """获取统计信息

        Returns:
            统计信息
        """
        await self.ensure_initialized()

        try:
            index_name = self.es_config.index_name or self.config.collection_name

            exists = await self._client.indices.exists(index=index_name)
            if not exists:
                return VectorStats(
                    total_vectors=0,
                    collections=0,
                    dimension=self.config.dimension,
                    metric=self.es_config.similarity,
                )

            # 获取索引统计
            stats = await self._client.indices.stats(index=index_name)
            total_vectors = stats["indices"][index_name]["primaries"]["docs"]["count"]

            return VectorStats(
                total_vectors=total_vectors,
                collections=1,
                dimension=self.config.dimension,
                metric=self.es_config.similarity,
            )

        except Exception as e:
            logger.error("elasticsearch_get_stats_failed", error=str(e))
            return VectorStats(
                total_vectors=0,
                collections=0,
                dimension=self.config.dimension,
                metric=self.es_config.similarity,
            )

    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client:
            await self._client.close()
            logger.info("elasticsearch_client_closed")


__all__ = [
    "ElasticsearchConfig",
    "ElasticsearchVectorStore",
]
