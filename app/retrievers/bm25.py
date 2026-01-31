"""BM25 检索器

使用 langchain_community 的 BM25 检索器，从文档构建关键词索引。
适用于关键词匹配场景，无需嵌入模型。

依赖安装:
    uv add langchain-community

使用示例:
```python
from app.retrievers import BM25Retriever, BM25RetrieverConfig

config = BM25RetrieverConfig(k=5)
retriever = BM25Retriever(config)
await retriever.from_texts(["文档1", "文档2"])
results = await retriever.aretrieve("查询内容")
```
"""

from dataclasses import dataclass
from typing import Any

from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BM25RetrieverConfig:
    """BM25 检索器配置

    Attributes:
        k: 返回结果数量
        score_threshold: 相似度阈值
        language: 语言 (zh/en)，用于分词
        tenant_id: 租户 ID
    """

    k: int = 5
    score_threshold: float | None = None
    language: str = "zh"
    tenant_id: int | None = None


class BM25Retriever(BaseRetriever):
    """BM25 检索器

    使用 BM25 算法进行关键词检索。
    支持中文分词（使用 jieba）。
    """

    config: BM25RetrieverConfig
    _inner_retriever: LC_BM25Retriever | None = None

    def __init__(
        self,
        config: BM25RetrieverConfig,
    ):
        """初始化 BM25 检索器

        Args:
            config: 检索器配置
        """
        super().__init__()
        self.config = config

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
        k: int | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """同步检索文档（LangChain 接口）

        Args:
            query: 查询文本
            run_manager: 运行管理器
            k: 返回结果数量
            score_threshold: 相似度阈值
            **kwargs: 额外参数

        Returns:
            文档列表
        """
        import asyncio

        return asyncio.run(
            self.aretrieve(
                query=query,
                k=k,
                score_threshold=score_threshold,
            )
        )

    async def aretrieve(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """异步检索文档

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            文档列表
        """
        if self._inner_retriever is None:
            logger.warning("bm25_retriever_not_initialized")
            return []

        k = k or self.config.k
        score_threshold = score_threshold or self.config.score_threshold

        # BM25 返回结果按分数排序
        docs = self._inner_retriever.get_relevant_documents(query)

        # 应用 k 限制
        results = docs[:k]

        # 应用阈值过滤
        if score_threshold is not None:
            # BM25 分数不是 0-1 范围，这里仅做简单过滤
            results = [d for d in results if self._get_score(d) >= score_threshold]

        logger.info(
            "bm25_retrieve_completed",
            query=query[:100],
            result_count=len(results),
        )

        return results

    def _get_score(self, document: Document) -> float:
        """从文档元数据获取分数

        Args:
            document: 文档对象

        Returns:
            分数
        """
        return document.metadata.get("score", 0.0)

    async def from_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """从文本列表构建索引

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        # 创建 BM25 检索器
        self._inner_retriever = LC_BM25Retriever.from_texts(
            texts=texts,
            metadatas=metadatas,
        )

        # 设置 k
        self._inner_retriever.k = self.config.k

        logger.info(
            "bm25_index_built",
            document_count=len(texts),
        )

    async def from_documents(
        self,
        documents: list[Document],
    ) -> None:
        """从文档列表构建索引

        Args:
            documents: 文档列表
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        await self.from_texts(texts, metadatas)

    def add_documents(self, documents: list[Document]) -> None:
        """添加文档到索引

        注意：需要重新构建索引。

        Args:
            documents: 文档列表
        """
        if self._inner_retriever is None:
            self._inner_retriever = LC_BM25Retriever.from_documents(documents)
        else:
            # 合并文档并重新构建
            existing_docs = self._inner_retriever.documents
            all_docs = existing_docs + documents
            self._inner_retriever = LC_BM25Retriever.from_documents(all_docs)

        self._inner_retriever.k = self.config.k

        logger.info(
            "bm25_documents_added",
            added_count=len(documents),
        )

    def delete_documents(self, document_ids: list[str]) -> None:
        """从索引中删除文档

        注意：需要重新构建索引。

        Args:
            document_ids: 文档 ID 列表
        """
        if self._inner_retriever is None:
            return

        # 过滤要删除的文档
        remaining_docs = [
            doc
            for doc in self._inner_retriever.documents
            if doc.metadata.get("id") not in document_ids
        ]

        self._inner_retriever = LC_BM25Retriever.from_documents(remaining_docs)
        self._inner_retriever.k = self.config.k

        logger.info(
            "bm25_documents_deleted",
            deleted_count=len(document_ids),
        )


__all__ = [
    "BM25Retriever",
    "BM25RetrieverConfig",
]
