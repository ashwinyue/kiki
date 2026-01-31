"""检索器模块

提供多种检索器功能：
- Elasticsearch: Elasticsearch 全文/向量检索
- BM25: BM25 关键词检索
- Ensemble: 多检索器融合
- Conversational: 对话式检索（带历史上下文）
"""

from app.retrievers.bm25 import (
    BM25Retriever,
    BM25RetrieverConfig,
)
from app.retrievers.conversational import (
    ConversationalRetriever,
    ConversationalRetrieverConfig,
    SearchResult,
)
from app.retrievers.elasticsearch import (
    ElasticsearchFilter,
    ElasticsearchRetriever,
    ElasticsearchRetrieverConfig,
    HighlightConfig,
    HybridSearchConfig,
)
from app.retrievers.ensemble import (
    EnsembleRetriever,
    EnsembleRetrieverConfig,
)

__all__ = [
    # BM25
    "BM25Retriever",
    "BM25RetrieverConfig",
    # Conversational
    "ConversationalRetriever",
    "ConversationalRetrieverConfig",
    "SearchResult",
    # Ensemble
    "EnsembleRetriever",
    "EnsembleRetrieverConfig",
    # Elasticsearch
    "ElasticsearchRetriever",
    "ElasticsearchRetrieverConfig",
    "ElasticsearchFilter",
    "HighlightConfig",
    "HybridSearchConfig",
]
