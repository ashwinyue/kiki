"""RAG 检索器模块

提供统一的检索器接口和多种后端实现。
"""

from app.agent.rag.retrievers.base import (
    BaseRetriever,
    RetrievedDocument,
    RetrievalError,
    RetrievalOptions,
)
from app.agent.rag.retrievers.faiss import FAISSRetriever
from app.agent.rag.retrievers.ragflow import (
    RAGFlowConfig,
    RAGFlowRetriever,
)

__all__ = [
    # 基础接口
    "BaseRetriever",
    "RetrievedDocument",
    "RetrievalError",
    "RetrievalOptions",
    # FAISS
    "FAISSRetriever",
    # RAGFlow
    "RAGFlowConfig",
    "RAGFlowRetriever",
]
