"""向量服务

提供向量存储的业务逻辑层。
"""

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config.settings import get_settings
from app.observability.logging import get_logger
from app.vector_stores import (
    BaseVectorStore,
    SearchResult,
    VectorStoreConfig,
    create_vector_store,
)

logger = get_logger(__name__)
settings = get_settings()


class VectorService:
    """向量服务

    提供向量索引、搜索、删除等功能。
    """

    def __init__(
        self,
        store: BaseVectorStore | None = None,
        embeddings: Embeddings | None = None,
        tenant_id: int | None = None,
    ):
        """初始化向量服务

        Args:
            store: 向量存储实例（默认从配置创建）
            embeddings: Embedding 实例
            tenant_id: 租户 ID
        """
        self.embeddings = embeddings
        self.tenant_id = tenant_id

        if store:
            self.store = store
        else:
            # 从配置创建
            config = VectorStoreConfig(tenant_id=tenant_id)
            store_type: str = getattr(
                settings, "vector_store_type", "memory"
            )  # type: ignore
            self.store = create_vector_store(store_type, config, embeddings)

    async def ensure_initialized(self) -> None:
        """确保向量存储已初始化"""
        if not self.store.is_initialized():
            await self.store.initialize()

    async def index_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
        collection_name: str = "default",
    ) -> str:
        """索引单个文档

        Args:
            content: 文档内容
            metadata: 元数据
            doc_id: 文档 ID
            collection_name: 集合名称

        Returns:
            文档 ID
        """
        await self.ensure_initialized()

        # 更新集合名称
        self.store.config.collection_name = collection_name

        document = Document(page_content=content, metadata=metadata or {})
        result = await self.store.add_documents([document], ids=[doc_id] if doc_id else None)

        logger.info(
            "document_indexed",
            doc_id=result.ids[0] if result.ids else None,
            collection=collection_name,
        )

        return result.ids[0] if result.ids else ""

    async def index_documents(
        self,
        documents: list[tuple[str, dict[str, Any] | None]],
        ids: list[str] | None = None,
        collection_name: str = "default",
    ) -> list[str]:
        """批量索引文档

        Args:
            documents: (content, metadata) 元组列表
            ids: 可选的文档 ID 列表
            collection_name: 集合名称

        Returns:
            文档 ID 列表
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        docs = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in documents
        ]

        result = await self.store.add_documents(docs, ids)

        logger.info(
            "documents_indexed",
            count=result.count,
            collection=collection_name,
        )

        return result.ids

    async def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
        collection_name: str = "default",
    ) -> tuple[list[SearchResult], int | None]:
        """向量搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值
            filter_dict: 过滤条件
            collection_name: 集合名称

        Returns:
            (搜索结果列表, 响应时间毫秒)
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        start_time = time.time()
        results = await self.store.similarity_search(query, k, score_threshold, filter_dict)
        response_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "vector_search_completed",
            query=query[:100],
            results_count=len(results),
            response_time_ms=response_time_ms,
            collection=collection_name,
        )

        return results, response_time_ms

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        filter_dict: dict[str, Any] | None = None,
        collection_name: str = "default",
    ) -> tuple[list[SearchResult], int | None]:
        """混合搜索（向量 + 关键词）

        Args:
            query: 查询文本
            k: 返回结果数量
            keyword_weight: 关键词搜索权重
            vector_weight: 向量搜索权重
            filter_dict: 过滤条件
            collection_name: 集合名称

        Returns:
            (搜索结果列表, 响应时间毫秒)
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        start_time = time.time()

        # 向量搜索
        vector_results = await self.store.similarity_search(
            query, k * 2, None, filter_dict
        )

        # 关键词搜索（简单实现：基于文本匹配）
        keyword_results = self._keyword_search(
            query, vector_results, filter_dict
        )

        # 合并结果
        results = self._merge_results(
            vector_results,
            keyword_results,
            vector_weight,
            keyword_weight,
            k,
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "hybrid_search_completed",
            query=query[:100],
            results_count=len(results),
            response_time_ms=response_time_ms,
            collection=collection_name,
        )

        return results, response_time_ms

    def _keyword_search(
        self,
        query: str,
        vector_results: list[SearchResult],
        filter_dict: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """关键词搜索

        简单实现：基于文本包含关系评分。
        """
        query_lower = query.lower()
        results = []

        for result in vector_results:
            # 应用过滤
            if filter_dict:
                if not all(
                    result.metadata.get(k) == v for k, v in filter_dict.items()
                ):
                    continue

            content_lower = result.content.lower()

            # 计算关键词匹配分数
            score = 0.0
            words = query_lower.split()

            for word in words:
                if word in content_lower:
                    score += 1.0

            if words:
                score = score / len(words)

            if score > 0:
                results.append(
                    SearchResult(
                        content=result.content,
                        metadata=result.metadata,
                        id=result.id,
                        score=score,
                    )
                )

        return results

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        vector_weight: float,
        keyword_weight: float,
        k: int,
    ) -> list[SearchResult]:
        """合并搜索结果"""
        merged: dict[str, SearchResult] = {}

        # 处理向量结果
        for result in vector_results:
            doc_id = result.id or result.content[:100]
            merged[doc_id] = SearchResult(
                content=result.content,
                metadata=result.metadata,
                id=result.id,
                score=result.score * vector_weight,
            )

        # 处理关键词结果
        for result in keyword_results:
            doc_id = result.id or result.content[:100]
            if doc_id in merged:
                # 合并分数
                merged[doc_id] = SearchResult(
                    content=merged[doc_id].content,
                    metadata=merged[doc_id].metadata,
                    id=merged[doc_id].id,
                    score=merged[doc_id].score + result.score * keyword_weight,
                )
            else:
                merged[doc_id] = SearchResult(
                    content=result.content,
                    metadata=result.metadata,
                    id=result.id,
                    score=result.score * keyword_weight,
                )

        # 排序并返回 top-k
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return results[:k]

    async def delete(
        self,
        doc_ids: list[str],
        collection_name: str = "default",
    ) -> bool:
        """删除文档

        Args:
            doc_ids: 文档 ID 列表
            collection_name: 集合名称

        Returns:
            是否成功
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        success = await self.store.delete(doc_ids)

        if success:
            logger.info(
                "documents_deleted",
                count=len(doc_ids),
                collection=collection_name,
            )

        return success

    async def delete_collection(
        self,
        collection_name: str = "default",
    ) -> bool:
        """删除整个集合

        Args:
            collection_name: 集合名称

        Returns:
            是否成功
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        success = await self.store.delete_collection()

        if success:
            logger.info("collection_deleted", collection=collection_name)

        return success

    async def get_stats(
        self,
        collection_name: str = "default",
    ) -> dict[str, Any]:
        """获取统计信息

        Args:
            collection_name: 集合名称

        Returns:
            统计信息字典
        """
        await self.ensure_initialized()

        self.store.config.collection_name = collection_name

        stats = await self.store.get_stats()

        return {
            "total_vectors": stats.total_vectors,
            "collections": stats.collections,
            "dimension": stats.dimension,
            "metric": stats.metric,
            "store_type": type(self.store).__name__,
        }

    async def health_check(self) -> bool:
        """健康检查

        Returns:
            是否健康
        """
        return await self.store.health_check()


__all__ = ["VectorService"]
