"""
向量存储模块 - FAISS 集成

提供向量数据库集成，支持语义检索和记忆管理。

参考：aold/ai-engineer-training2/week01/code/05-2langgraph.py
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.observability.logging import get_logger

logger = get_logger(__name__)

# 尝试导入 FAISS（可选依赖）
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss_not_available", message="FAISS 未安装，向量存储功能将不可用")


# 默认嵌入模型
_default_embeddings: Embeddings | None = None


def get_default_embeddings() -> Embeddings:
    """获取默认嵌入模型

    Returns:
        嵌入模型实例

    Raises:
        RuntimeError: 如果嵌入模型未配置
    """
    global _default_embeddings

    if _default_embeddings is None:
        from app.llm.embeddings import get_embedding_model
        _default_embeddings = get_embedding_model()

    return _default_embeddings


class VectorStoreManager:
    """向量存储管理器

    管理向量数据库的创建、检索和更新。
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        store_type: str = "faiss",
    ):
        """初始化向量存储管理器

        Args:
            embeddings: 嵌入模型（默认使用系统默认）
            store_type: 存储类型（目前仅支持 faiss）

        Raises:
            ValueError: 如果指定的存储类型不可用
        """
        if store_type == "faiss" and not FAISS_AVAILABLE:
            raise ValueError("FAISS 不可用，请安装 langchain-community")

        self.embeddings = embeddings or get_default_embeddings()
        self.store_type = store_type
        self._store: VectorStore | None = None

        logger.info(
            "vector_store_manager_initialized",
            store_type=store_type,
        )

    def create_from_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> VectorStore:
        """从文本列表创建向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）

        Returns:
            向量存储实例

        Raises:
            RuntimeError: 如果创建失败
        """
        try:
            if self.store_type == "faiss":
                self._store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                )
                logger.info(
                    "faiss_store_created",
                    doc_count=len(texts),
                )
                return self._store
            else:
                raise ValueError(f"不支持的存储类型: {self.store_type}")

        except Exception as e:
            logger.error("vector_store_creation_failed", error=str(e))
            raise RuntimeError(f"向量存储创建失败: {e}") from e

    def create_from_documents(
        self,
        documents: list[Document],
    ) -> VectorStore:
        """从文档列表创建向量存储

        Args:
            documents: 文档列表

        Returns:
            向量存储实例
        """
        try:
            if self.store_type == "faiss":
                self._store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                )
                logger.info(
                    "faiss_store_created_from_docs",
                    doc_count=len(documents),
                )
                return self._store
            else:
                raise ValueError(f"不支持的存储类型: {self.store_type}")

        except Exception as e:
            logger.error("vector_store_creation_failed", error=str(e))
            raise RuntimeError(f"向量存储创建失败: {e}") from e

    def add_texts(
        self,
        texts: list[str],
    ) -> None:
        """向现有存储添加文本

        Args:
            texts: 文本列表

        Raises:
            RuntimeError: 如果存储未初始化
        """
        if self._store is None:
            raise RuntimeError("向量存储未初始化")

        try:
            if isinstance(self._store, FAISS):
                self._store.add_texts(texts=texts)
                logger.info(
                    "texts_added_to_store",
                    count=len(texts),
                )
            else:
                raise ValueError("不支持的存储类型")

        except Exception as e:
            logger.error("add_texts_failed", error=str(e))
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件（可选）

        Returns:
            相关文档列表

        Raises:
            RuntimeError: 如果存储未初始化
        """
        if self._store is None:
            raise RuntimeError("向量存储未初始化")

        try:
            results = self._store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
            logger.debug(
                "similarity_search_completed",
                query=query[:50],
                result_count=len(results),
            )
            return results

        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """相似度搜索（带分数）

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件（可选）

        Returns:
            (文档, 分数) 元组列表
        """
        if self._store is None:
            raise RuntimeError("向量存储未初始化")

        try:
            results = self._store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )
            return results

        except Exception as e:
            logger.error("similarity_search_with_score_failed", error=str(e))
            return []

    def get_store(self) -> VectorStore | None:
        """获取向量存储实例

        Returns:
            向量存储实例（可能为 None）
        """
        return self._store

    def clear(self) -> None:
        """清空向量存储"""
        self._store = None
        logger.info("vector_store_cleared")


def retrieve_relevant_context(
    query: str,
    vectorstore: VectorStore | None,
    k: int = 3,
    score_threshold: float | None = None,
) -> str:
    """检索相关上下文

    Args:
        query: 查询文本
        vectorstore: 向量存储实例
        k: 返回结果数量
        score_threshold: 相似度分数阈值（可选）

    Returns:
        拼接的相关上下文文本
    """
    if vectorstore is None:
        logger.warning("vectorstore_not_available")
        return ""

    try:
        if score_threshold is not None:
            # 使用带分数的搜索
            results_with_scores = vectorstore.similarity_search_with_score(
                query=query,
                k=k,
            )
            # 过滤低分结果
            filtered_results = [
                doc for doc, score in results_with_scores
                if score >= score_threshold
            ]
            context_parts = [doc.page_content for doc in filtered_results]
        else:
            results = vectorstore.similarity_search(query=query, k=k)
            context_parts = [doc.page_content for doc in results]

        context = "\n\n".join(context_parts)

        logger.debug(
            "context_retrieved",
            query=query[:50],
            char_count=len(context),
            source_count=len(context_parts),
        )

        return context

    except Exception as e:
        logger.error("context_retrieval_failed", error=str(e))
        return ""


def create_document_store(
    texts: list[str],
    embeddings: Embeddings | None = None,
) -> VectorStore | None:
    """便捷函数：创建文档向量存储

    Args:
        texts: 文本列表
        embeddings: 嵌入模型（可选）

    Returns:
        向量存储实例（失败返回 None）
    """
    if not FAISS_AVAILABLE:
        logger.warning("faiss_not_available_cannot_create_store")
        return None

    try:
        manager = VectorStoreManager(embeddings=embeddings)
        return manager.create_from_texts(texts=texts)

    except Exception as e:
        logger.error("document_store_creation_failed", error=str(e))
        return None


# 导出的便捷函数
__all__ = [
    "VectorStoreManager",
    "retrieve_relevant_context",
    "create_document_store",
    "get_default_embeddings",
    "FAISS_AVAILABLE",
]
