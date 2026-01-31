"""Elasticsearch 检索器

封装 Elasticsearch 检索功能，支持高级查询、高亮、过滤等。

依赖安装:
    uv add langchain-elasticsearch "elasticsearch[async]>=8.0.0"

使用示例:
```python
from app.retrievers import ElasticsearchRetriever, ElasticsearchRetrieverConfig

config = ElasticsearchRetrieverConfig(
    index_name="documents",
    es_url="http://localhost:9200",
)

retriever = ElasticsearchRetriever(config)
results = await retriever.retrieve("查询内容", k=5)
```
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HighlightConfig:
    """高亮配置

    Attributes:
        enabled: 是否启用高亮
        pre_tags: 前置标签
        post_tags: 后置标签
        fragment_size: 片段大小
        number_of_fragments: 返回片段数量
    """

    enabled: bool = True
    pre_tags: list[str] = field(default_factory=lambda: ["<em>"])
    post_tags: list[str] = field(default_factory=lambda: ["</em>"])
    fragment_size: int = 150
    number_of_fragments: int = 3


@dataclass
class HybridSearchConfig:
    """混合搜索配置

    Attributes:
        enabled: 是否启用混合搜索
        text_weight: 文本搜索权重
        vector_weight: 向量搜索权重
        rerank: 是否重排序
        rank_constant: RRF 常数 (Reciprocal Rank Fusion)
    """

    enabled: bool = False
    text_weight: float = 0.5
    vector_weight: float = 0.5
    rerank: bool = False
    rank_constant: int = 60


@dataclass
class ElasticsearchFilter:
    """Elasticsearch 过滤器

    Attributes:
        term: 精确匹配 {field: value}
        terms: 多值匹配 {field: [values]}
        range: 范围查询 {field: {gt/gte/lt/lte: value}}
        exists: 字段存在检查
        prefix: 前缀匹配
        wildcard: 通配符匹配
        geo_distance: 地理距离查询
    """

    term: dict[str, Any] = field(default_factory=dict)
    terms: dict[str, list[Any]] = field(default_factory=dict)
    range: dict[str, dict[str, Any]] = field(default_factory=dict)
    exists: list[str] = field(default_factory=list)
    prefix: dict[str, str] = field(default_factory=dict)
    wildcard: dict[str, str] = field(default_factory=dict)
    geo_distance: dict[str, Any] = field(default_factory=dict)

    def build_query(self) -> dict[str, Any] | None:
        """构建过滤查询 DSL

        Returns:
            Elasticsearch 查询字典
        """
        must = []

        for field_name, value in self.term.items():
            must.append({"term": {field_name: value}})

        for field_name, values in self.terms.items():
            must.append({"terms": {field_name: values}})

        for field_name, range_value in self.range.items():
            must.append({"range": {field_name: range_value}})

        for field_name in self.exists:
            must.append({"exists": {"field": field_name}})

        for field_name, prefix in self.prefix.items():
            must.append({"prefix": {field_name: prefix}})

        for field_name, pattern in self.wildcard.items():
            must.append({"wildcard": {field_name: pattern}})

        if self.geo_distance:
            location_field = self.geo_distance.get("location_field", "location")
            must.append(
                {
                    "geo_distance": {
                        "distance": self.geo_distance.get("distance", "10km"),
                        location_field: {
                            "lat": self.geo_distance.get("lat"),
                            "lon": self.geo_distance.get("lon"),
                        },
                    }
                }
            )

        return {"bool": {"must": must}} if must else None


@dataclass
class ElasticsearchRetrieverConfig:
    """Elasticsearch 检索器配置

    Attributes:
        es_url: Elasticsearch URL
        cloud_id: Elastic Cloud ID
        api_key: API Key
        username: 用户名
        password: 密码
        index_name: 索引名称
        default_k: 默认返回结果数量
        score_threshold: 相似度阈值
        highlight_config: 高亮配置
        hybrid_config: 混合搜索配置
        verify_certs: 是否验证证书
        request_timeout: 请求超时时间
        tenant_id: 租户 ID
    """

    # 连接配置
    es_url: str | None = "http://localhost:9200"
    cloud_id: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None

    # 索引配置
    index_name: str = "documents"
    vector_field: str = "vector"
    text_field: str = "text"
    metadata_field: str = "metadata"

    # 搜索配置
    default_k: int = 5
    score_threshold: float | None = None
    min_score: float | None = None

    # 高级配置
    highlight_config: HighlightConfig = field(default_factory=HighlightConfig)
    hybrid_config: HybridSearchConfig = field(default_factory=HybridSearchConfig)

    # 连接选项
    verify_certs: bool = True
    request_timeout: int = 30
    max_retries: int = 3

    # 多租户
    tenant_id: int | None = None


class ElasticsearchRetriever(BaseRetriever):
    """Elasticsearch 检索器

    支持多种检索模式：
    - 关键词检索 (BM25)
    - 向量检索 (Dense Vector)
    - 混合检索 (Hybrid)
    """

    config: ElasticsearchRetrieverConfig
    embeddings: Embeddings | None = None
    _client: Any = None

    def __init__(
        self,
        config: ElasticsearchRetrieverConfig,
        embeddings: Embeddings | None = None,
    ):
        """初始化检索器

        Args:
            config: 检索器配置
            embeddings: Embedding 实例（向量搜索需要）
        """
        super().__init__()
        self.config = config
        self.embeddings = embeddings

    async def _initialize(self) -> None:
        """初始化客户端"""
        if self._client is not None:
            return

        try:
            from elasticsearch import AsyncElasticsearch

            params = self._build_client_params()
            self._client = AsyncElasticsearch(**params)

            # 测试连接
            await self._client.ping()
            logger.info("elasticsearch_retriever_initialized")

        except Exception as e:
            logger.error("elasticsearch_retriever_init_failed", error=str(e))
            raise

    def _build_client_params(self) -> dict[str, Any]:
        """构建客户端参数"""
        params: dict[str, Any] = {
            "verify_certs": self.config.verify_certs,
            "request_timeout": self.config.request_timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.cloud_id:
            params["cloud_id"] = self.config.cloud_id
        elif self.config.es_url:
            params["hosts"] = [self.config.es_url]

        if self.config.api_key:
            params["api_key"] = self.config.api_key
        elif self.config.username and self.config.password:
            params["basic_auth"] = (self.config.username, self.config.password)

        return params

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
        k: int | None = None,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
        search_type: Literal["keyword", "vector", "hybrid"] = "hybrid",
        **kwargs: Any,
    ) -> list[Document]:
        """同步检索文档（LangChain 接口）

        Args:
            query: 查询文本
            run_manager: 运行管理器
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件
            search_type: 搜索类型
            **kwargs: 额外参数

        Returns:
            文档列表
        """
        # 注意：这是同步方法，需要使用 asyncio 运行
        import asyncio

        return asyncio.run(
            self.aretrieve(
                query=query,
                k=k or self.config.default_k,
                score_threshold=score_threshold,
                filter_dict=filter_dict,
                search_type=search_type,
            )
        )

    async def aretrieve(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
        search_type: Literal["keyword", "vector", "hybrid"] = "hybrid",
        enable_highlight: bool = True,
    ) -> list[Document]:
        """异步检索文档

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件
            search_type: 搜索类型
            enable_highlight: 是否启用高亮

        Returns:
            文档列表
        """
        await self._initialize()

        k = k or self.config.default_k
        score_threshold = score_threshold or self.config.score_threshold

        try:
            if search_type == "keyword":
                return await self._keyword_search(
                    query,
                    k,
                    score_threshold,
                    filter_dict,
                    enable_highlight,
                )
            elif search_type == "vector":
                return await self._vector_search(
                    query,
                    k,
                    score_threshold,
                    filter_dict,
                )
            else:
                return await self._hybrid_search(
                    query,
                    k,
                    score_threshold,
                    filter_dict,
                    enable_highlight,
                )

        except Exception as e:
            logger.error("elasticsearch_retrieve_failed", error=str(e))
            return []

    async def _keyword_search(
        self,
        query: str,
        k: int,
        score_threshold: float | None,
        filter_dict: dict[str, Any] | None,
        enable_highlight: bool,
    ) -> list[Document]:
        """关键词搜索 (BM25)

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件
            enable_highlight: 是否启用高亮

        Returns:
            文档列表
        """
        search_body: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                self.config.text_field: {
                                    "query": query,
                                }
                            }
                        }
                    ]
                }
            },
        }

        # 添加过滤
        if filter_dict or self.config.tenant_id:
            search_body["query"]["bool"]["filter"] = self._build_filters(filter_dict)

        # 添加高亮
        if enable_highlight and self.config.highlight_config.enabled:
            search_body["highlight"] = {
                "pre_tags": self.config.highlight_config.pre_tags,
                "post_tags": self.config.highlight_config.post_tags,
                "fields": {
                    self.config.text_field: {
                        "fragment_size": self.config.highlight_config.fragment_size,
                        "number_of_fragments": self.config.highlight_config.number_of_fragments,
                    }
                },
            }

        # 添加最小分数
        if self.config.min_score:
            search_body["min_score"] = self.config.min_score

        # 执行搜索
        response = await self._client.search(
            index=self.config.index_name,
            body=search_body,
        )

        # 转换结果
        documents = []
        for hit in response["hits"]["hits"]:
            score = hit.get("_score", 0.0)
            if score_threshold and score < score_threshold:
                continue

            source = hit["_source"]
            metadata = {
                "score": score,
                "id": hit.get("_id"),
                **source.get(self.config.metadata_field, {}),
            }

            # 添加高亮
            if enable_highlight and "highlight" in hit:
                highlights = hit["highlight"].get(self.config.text_field, [])
                metadata["highlights"] = highlights

            documents.append(
                Document(
                    page_content=source.get(self.config.text_field, ""),
                    metadata=metadata,
                )
            )

        return documents

    async def _vector_search(
        self,
        query: str,
        k: int,
        score_threshold: float | None,
        filter_dict: dict[str, Any] | None,
    ) -> list[Document]:
        """向量搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件

        Returns:
            文档列表
        """
        if not self.embeddings:
            raise ValueError("Embeddings required for vector search")

        # 嵌入查询
        query_vector = await self.embeddings.aembed_query(query)

        search_body: dict[str, Any] = {
            "size": k,
            "knn": {
                "field": self.config.vector_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10,
            },
        }

        # 添加过滤
        if filter_dict or self.config.tenant_id:
            search_body["query"] = {
                "bool": {
                    "filter": self._build_filters(filter_dict),
                }
            }

        # 添加最小分数
        if self.config.min_score:
            search_body["min_score"] = self.config.min_score

        # 执行搜索
        response = await self._client.search(
            index=self.config.index_name,
            body=search_body,
        )

        # 转换结果
        documents = []
        for hit in response["hits"]["hits"]:
            score = hit.get("_score", 0.0)
            if score_threshold and score < score_threshold:
                continue

            source = hit["_source"]
            metadata = {
                "score": score,
                "id": hit.get("_id"),
                **source.get(self.config.metadata_field, {}),
            }

            documents.append(
                Document(
                    page_content=source.get(self.config.text_field, ""),
                    metadata=metadata,
                )
            )

        return documents

    async def _hybrid_search(
        self,
        query: str,
        k: int,
        score_threshold: float | None,
        filter_dict: dict[str, Any] | None,
        enable_highlight: bool,
    ) -> list[Document]:
        """混合搜索 (RRF)

        结合关键词搜索和向量搜索，使用 Reciprocal Rank Fusion 合并结果。

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件
            enable_highlight: 是否启用高亮

        Returns:
            文档列表
        """
        if not self.embeddings:
            logger.warning("embeddings_not_configured_fallback_to_keyword")
            return await self._keyword_search(
                query, k, score_threshold, filter_dict, enable_highlight
            )

        # 构建混合查询
        query_vector = await self.embeddings.aembed_query(query)

        hybrid_body: dict[str, Any] = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                self.config.text_field: {
                                    "query": query,
                                    "boost": self.config.hybrid_config.text_weight,
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": self.config.vector_field,
                                "query_vector": query_vector,
                                "k": k,
                                "boost": self.config.hybrid_config.vector_weight,
                            }
                        },
                    ]
                }
            },
        }

        # 添加过滤
        if filter_dict or self.config.tenant_id:
            hybrid_body["query"]["bool"]["filter"] = self._build_filters(filter_dict)

        # 添加高亮
        if enable_highlight and self.config.highlight_config.enabled:
            hybrid_body["highlight"] = {
                "pre_tags": self.config.highlight_config.pre_tags,
                "post_tags": self.config.highlight_config.post_tags,
                "fields": {
                    self.config.text_field: {
                        "fragment_size": self.config.highlight_config.fragment_size,
                        "number_of_fragments": self.config.highlight_config.number_of_fragments,
                    }
                },
            }

        # 添加最小分数
        if self.config.min_score:
            hybrid_body["min_score"] = self.config.min_score

        # 执行搜索
        response = await self._client.search(
            index=self.config.index_name,
            body=hybrid_body,
        )

        # 转换结果
        documents = []
        for hit in response["hits"]["hits"]:
            score = hit.get("_score", 0.0)
            if score_threshold and score < score_threshold:
                continue

            source = hit["_source"]
            metadata = {
                "score": score,
                "id": hit.get("_id"),
                **source.get(self.config.metadata_field, {}),
            }

            # 添加高亮
            if enable_highlight and "highlight" in hit:
                highlights = hit["highlight"].get(self.config.text_field, [])
                metadata["highlights"] = highlights

            documents.append(
                Document(
                    page_content=source.get(self.config.text_field, ""),
                    metadata=metadata,
                )
            )

        return documents

    def _build_filters(self, filter_dict: dict[str, Any] | None) -> list[dict[str, Any]]:
        """构建过滤条件

        Args:
            filter_dict: 过滤字典

        Returns:
            Elasticsearch 过滤列表
        """
        filters = []

        # 租户过滤
        if self.config.tenant_id:
            filters.append({"term": {"tenant_id": self.config.tenant_id}})

        # 额外过滤
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, (list, tuple)):
                    filters.append({"terms": {key: list(value)}})
                elif isinstance(value, dict):
                    # 支持复杂查询
                    filters.append(value)
                else:
                    filters.append({"term": {key: value}})

        return filters

    async def close(self) -> None:
        """关闭客户端连接"""
        if self._client:
            await self._client.close()
            logger.info("elasticsearch_retriever_closed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


__all__ = [
    "ElasticsearchRetriever",
    "ElasticsearchRetrieverConfig",
    "ElasticsearchFilter",
    "HighlightConfig",
    "HybridSearchConfig",
]
