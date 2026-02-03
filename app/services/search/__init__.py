"""搜索模块

提供基础关键词搜索功能（Elasticsearch 可选）。

模块结构:
- elasticsearch: Elasticsearch 客户端封装（可选）
- indexer: 文档索引器（可选）
- query: 查询构建器（可选）
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
]
