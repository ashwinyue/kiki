"""Agent 能力模块

提供 RAG、澄清等 Agent 高级能力。
"""

from app.agent.capabilities.clarification import (
    ClarificationNode,
    build_clarified_query,
    build_clarified_topic_from_history,
    complete_clarification,
    create_clarification_prompt,
    format_clarification_context,
    get_clarification_summary,
    needs_clarification,
    record_clarification,
    reset_clarification,
    should_prompt_clarification,
)
from app.agent.capabilities.rag import (
    BaseVectorStore,
    ChromaStore,
    PgVectorStore,
    PineconeStore,
    SearchResult,
    VectorStoreConfig,
    VectorStoreType,
    create_vector_store,
    index_documents,
    retrieve_documents,
)

__all__ = [
    # RAG
    "BaseVectorStore",
    "PgVectorStore",
    "PineconeStore",
    "ChromaStore",
    "VectorStoreConfig",
    "VectorStoreType",
    "SearchResult",
    "create_vector_store",
    "retrieve_documents",
    "index_documents",
    # Clarification
    "ClarificationNode",
    "needs_clarification",
    "should_prompt_clarification",
    "record_clarification",
    "complete_clarification",
    "reset_clarification",
    "build_clarified_query",
    "build_clarified_topic_from_history",
    "get_clarification_summary",
    "create_clarification_prompt",
    "format_clarification_context",
]
