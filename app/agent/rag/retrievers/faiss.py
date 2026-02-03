"""FAISS 本地检索器

基于 LangChain FAISS 的本地向量检索器。
复用已有的 app.agent.vector_store 模块。
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.agent.vector_store import (
    VectorStoreManager,
    get_default_embeddings,
)
from app.observability.logging import get_logger
from app.agent.rag.retrievers.base import (
    BaseRetriever,
    RetrievedDocument,
    RetrievalOptions,
    RetrievalError,
)

logger = get_logger(__name__)


class FAISSRetriever(BaseRetriever):
    """FAISS 本地向量检索器

    使用 FAISS 进行本地向量存储和语义检索。

    Attributes:
        embeddings: 嵌入模型（可选，默认使用系统默认）
        store_manager: 向量存储管理器

    Example:
        ```python
        # 创建检索器
        retriever = FAISSRetriever()

        # 添加文档
        retriever.add_texts([
            "Python 是一种高级编程语言",
            "JavaScript 主要用于 Web 开发",
        ])

        # 检索
        results = await retriever.retrieve("编程语言")
        ```
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        store_type: str = "faiss",
    ):
        """初始化 FAISS 检索器

        Args:
            embeddings: 嵌入模型（可选，默认使用系统默认）
            store_type: 存储类型（默认 faiss）

        Raises:
            ValueError: 如果 FAISS 不可用
        """
        self.embeddings = embeddings or get_default_embeddings()
        self.store_manager = VectorStoreManager(
            embeddings=self.embeddings,
            store_type=store_type,
        )

        logger.info(
            "faiss_retriever_initialized",
            store_type=store_type,
        )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）

        Raises:
            RetrievalError: 添加失败时抛出
        """
        try:
            self.store_manager.create_from_texts(
                texts=texts,
                metadatas=metadatas,
            )
            logger.info(
                "texts_added_to_faiss",
                count=len(texts),
            )
        except Exception as e:
            raise RetrievalError(
                message=f"添加文本失败: {e}",
                retriever_type="FAISS",
                cause=e,
            )

    def add_documents(self, documents: list[Document]) -> None:
        """添加文档到向量存储

        Args:
            documents: 文档列表

        Raises:
            RetrievalError: 添加失败时抛出
        """
        try:
            self.store_manager.create_from_documents(documents=documents)
            logger.info(
                "documents_added_to_faiss",
                count=len(documents),
            )
        except Exception as e:
            raise RetrievalError(
                message=f"添加文档失败: {e}",
                retriever_type="FAISS",
                cause=e,
            )

    async def retrieve(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedDocument]:
        """检索相关文档

        Args:
            query: 查询文本
            options: 检索选项（可选）

        Returns:
            检索到的文档列表

        Raises:
            RetrievalError: 检索失败时抛出
        """
        if options is None:
            options = RetrievalOptions()

        try:
            # 使用带分数的检索
            results_with_scores = self.store_manager.similarity_search_with_score(
                query=query,
                k=options.top_k,
                filter=options.filter_kwargs,
            )

            # 过滤低分结果
            filtered_results = [
                (doc, score)
                for doc, score in results_with_scores
                if options.score_threshold is None or score >= options.score_threshold
            ]

            # 转换为 RetrievedDocument
            retrieved_docs = []
            for doc, score in filtered_results:
                retrieved_docs.append(
                    RetrievedDocument(
                        title=doc.metadata.get("title", "Untitled"),
                        content=doc.page_content,
                        source=doc.metadata.get("source", "unknown"),
                        score=float(score),
                        metadata=doc.metadata,
                    )
                )

            logger.debug(
                "faiss_retrieval_completed",
                query=query[:50],
                result_count=len(retrieved_docs),
            )

            return retrieved_docs

        except Exception as e:
            logger.error(
                "faiss_retrieval_failed",
                query=query[:50],
                error=str(e),
            )
            raise RetrievalError(
                message=f"检索失败: {e}",
                retriever_type="FAISS",
                cause=e,
            )

    def retrieve_sync(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedDocument]:
        """同步检索（FAISS 原生支持，无需异步包装）

        Args:
            query: 查询文本
            options: 检索选项（可选）

        Returns:
            检索到的文档列表
        """
        import asyncio

        return asyncio.run(self.retrieve(query, options))

    def health_check(self) -> bool:
        """健康检查

        Returns:
            向量存储是否可用
        """
        try:
            store = self.store_manager.get_store()
            # 检查存储是否已初始化
            return store is not None
        except Exception as e:
            logger.warning(
                "faiss_health_check_failed",
                error=str(e),
            )
            return False

    def clear(self) -> None:
        """清空向量存储"""
        self.store_manager.clear()
        logger.info("faiss_store_cleared")


__all__ = ["FAISSRetriever"]
