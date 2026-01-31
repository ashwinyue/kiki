"""搜索模块

提供 Elasticsearch 全文搜索和混合检索功能。

模块结构:
- elasticsearch: Elasticsearch 客户端封装
- indexer: 文档索引器
- query: 查询构建器
- service: 增强搜索服务 (WeKnora99 对齐)
"""

from app.services.search.elasticsearch import (
    ElasticsearchClient,
    close_elasticsearch_client,
    elasticsearch_context,
    get_elasticsearch_client,
)
from app.services.search.indexer import DocumentIndexer, create_document_indexer
from app.services.search.query import (
    AggregationBucket,
    QueryBuilder,
    SearchResponse,
    SearchResult,
    parse_date_histogram_aggregation,
    parse_terms_aggregation,
)
from app.services.search.service import (
    KeywordSearcher,
    MemoryVectorSearcher,
    QdrantVectorSearcher,
    SearchResultPostProcessor,
    SearchResultType,
    SearchService,
    VectorSearcher,
    hybrid_search,
)

__all__ = [
    # Elasticsearch 客户端
    "ElasticsearchClient",
    "get_elasticsearch_client",
    "close_elasticsearch_client",
    "elasticsearch_context",
    # 文档索引
    "DocumentIndexer",
    "create_document_indexer",
    # 查询构建
    "QueryBuilder",
    "SearchResult",
    "SearchResponse",
    "AggregationBucket",
    "parse_terms_aggregation",
    "parse_date_histogram_aggregation",
    # 增强搜索服务
    "SearchService",
    "hybrid_search",
    "SearchResultType",
    "SearchResultPostProcessor",
    "VectorSearcher",
    "QdrantVectorSearcher",
    "MemoryVectorSearcher",
    "KeywordSearcher",
]
