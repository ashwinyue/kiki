"""增强搜索服务

对齐 WeKnora99 + DeerFlow 的混合搜索架构：
- 向量搜索 (Qdrant/Pinecone/Memory)
- 关键词搜索 (PostgreSQL 全文搜索)
- 分数归一化与合并
- 搜索结果后处理
- MMR 多样性重排序

使用示例:
    ```python
    from app.services.search import SearchService

    search_service = SearchService(session)
    results = await search_service.hybrid_search(
        knowledge_base_id="kb-1",
        query="什么是 Python?",
        top_k=20,
    )
    ```
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.repositories.knowledge import ChunkRepository
from app.vector_stores import VectorStoreConfig, create_vector_store

logger = get_logger(__name__)


class SearchResultType(str, Enum):
    """搜索结果类型"""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    DIRECT_LOAD = "direct_load"
    HISTORY = "history"
    WEB_SEARCH = "web_search"


@dataclass
class SearchResult:
    """搜索结果

    对齐 DeerFlow Document + WeKnora99 SearchResult
    """

    id: str  # Chunk ID
    content: str
    score: float
    knowledge_id: str | None = None
    knowledge_title: str | None = None
    knowledge_filename: str | None = None
    knowledge_source: str | None = None
    chunk_index: int | None = None
    match_type: SearchResultType = SearchResultType.HYBRID
    chunk_type: str | None = None
    parent_chunk_id: str | None = None
    image_info: str | None = None  # JSON string
    start_at: int | None = None
    end_at: int | None = None
    url: str | None = None  # Web search URL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "knowledge_id": self.knowledge_id,
            "knowledge_title": self.knowledge_title,
            "knowledge_filename": self.knowledge_filename,
            "knowledge_source": self.knowledge_source,
            "chunk_index": self.chunk_index,
            "match_type": self.match_type.value,
            "chunk_type": self.chunk_type,
            "url": self.url,
            "metadata": self.metadata,
        }


class SearchResultPostProcessor:
    """搜索结果后处理器

    对齐 DeerFlow SearchResultPostProcessor
    """

    def __init__(
        self,
        min_score_threshold: float = 0.0,
        max_content_length: int = 4000,
    ):
        self.min_score_threshold = min_score_threshold
        self.max_content_length = max_content_length
        self.base64_pattern = re.compile(
            r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+"
        )

    async def process(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """处理搜索结果

        Args:
            results: 原始搜索结果

        Returns:
            处理后的结果
        """
        if not results:
            return []

        cleaned = []
        seen = set()

        for r in results:
            # 1. 去重
            key = self._get_dedup_key(r)
            if key in seen:
                continue
            seen.add(key)

            # 2. 过滤低分结果
            if r.score < self.min_score_threshold:
                continue

            # 3. 清理 base64 图片
            r.content = self._remove_base64_images(r.content)

            # 4. 截断过长内容
            if len(r.content) > self.max_content_length:
                r.content = r.content[: self.max_content_length] + "..."

            if r.content:
                cleaned.append(r)

        # 5. 按分数排序
        cleaned.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "search_post_process",
            input_count=len(results),
            output_count=len(cleaned),
        )

        return cleaned

    def _get_dedup_key(self, result: SearchResult) -> str:
        """获取去重键"""
        if result.url:
            return result.url
        if result.parent_chunk_id:
            return result.parent_chunk_id
        return result.id

    def _remove_base64_images(self, content: str) -> str:
        """移除 base64 编码的图片"""
        return self.base64_pattern.sub(" ", content)


class VectorSearcher(ABC):
    """向量搜索器抽象接口"""

    @abstractmethod
    async def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        score_threshold: float,
        knowledge_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """执行向量搜索"""
        pass


class QdrantVectorSearcher(VectorSearcher):
    """Qdrant 向量搜索器

    对齐 DeerFlow QdrantProvider
    """

    def __init__(
        self,
        tenant_id: int,
        embedding_model_id: str | None = None,
    ):
        self.tenant_id = tenant_id
        self.embedding_model_id = embedding_model_id
        self._store = None

    def _get_store(self, kb_id: str) -> Any:
        """获取向量存储实例"""
        if self._store is None:
            config = VectorStoreConfig(
                collection_name=f"kb_{kb_id}",
                tenant_id=self.tenant_id,
            )
            self._store = create_vector_store("qdrant", config)
        return self._store

    async def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        score_threshold: float,
        knowledge_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """执行向量搜索"""
        try:
            store = self._get_store(knowledge_base_id)
            await store.initialize()

            # 构建过滤器
            filter_dict = {}
            if knowledge_ids:
                filter_dict["knowledge_id"] = knowledge_ids

            results = await store.search(
                query=query,
                k=top_k,
                score_threshold=score_threshold,
                filter_dict=filter_dict,
            )

            return [
                SearchResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    knowledge_id=r.get("metadata", {}).get("knowledge_id"),
                    knowledge_title=r.get("metadata", {}).get("knowledge_title"),
                    knowledge_filename=r.get("metadata", {}).get("knowledge_filename"),
                    chunk_index=r.get("metadata", {}).get("chunk_index"),
                    match_type=SearchResultType.VECTOR,
                    chunk_type=r.get("metadata", {}).get("chunk_type"),
                    parent_chunk_id=r.get("metadata", {}).get("parent_chunk_id"),
                    start_at=r.get("metadata", {}).get("start_at"),
                    end_at=r.get("metadata", {}).get("end_at"),
                )
                for r in results
            ]

        except Exception as e:
            logger.warning(
                "vector_search_failed",
                kb_id=knowledge_base_id,
                error=str(e),
            )
            return []


class MemoryVectorSearcher(VectorSearcher):
    """内存向量搜索器（用于开发/测试）"""

    def __init__(self, tenant_id: int):
        self.tenant_id = tenant_id
        self._store = None

    def _get_store(self, kb_id: str) -> Any:
        """获取向量存储实例"""
        if self._store is None:
            config = VectorStoreConfig(
                collection_name=f"kb_{kb_id}",
                tenant_id=self.tenant_id,
            )
            self._store = create_vector_store("memory", config)
        return self._store

    async def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        score_threshold: float,
        knowledge_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """执行向量搜索"""
        try:
            store = self._get_store(knowledge_base_id)
            await store.initialize()

            results = await store.search(
                query=query,
                k=top_k,
                score_threshold=score_threshold,
            )

            return [
                SearchResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    match_type=SearchResultType.VECTOR,
                )
                for r in results
            ]

        except Exception as e:
            logger.warning(
                "memory_vector_search_failed",
                kb_id=knowledge_base_id,
                error=str(e),
            )
            return []


class KeywordSearcher:
    """关键词搜索器

    使用 PostgreSQL 全文搜索 (tsvector)
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._chunk_repo: ChunkRepository | None = None

    @property
    def chunk_repo(self) -> ChunkRepository:
        if self._chunk_repo is None:
            self._chunk_repo = ChunkRepository(self.session)
        return self._chunk_repo

    async def search(
        self,
        query: str,
        knowledge_base_id: str,
        tenant_id: int,
        top_k: int,
        score_threshold: float,
        knowledge_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """执行关键词搜索"""
        from app.models.knowledge import Chunk, Knowledge

        try:
            # 构建基础查询
            stmt = (
                select(Chunk)
                .join(Knowledge, Chunk.knowledge_id == Knowledge.id)
                .where(
                    Chunk.knowledge_base_id == knowledge_base_id,
                    Chunk.tenant_id == tenant_id,
                    Chunk.is_enabled == True,
                    Chunk.deleted_at.is_(None),
                    Knowledge.deleted_at.is_(None),
                    Knowledge.parse_status == "completed",
                )
            )

            # 添加知识 ID 过滤
            if knowledge_ids:
                stmt = stmt.where(Chunk.knowledge_id.in_(knowledge_ids))

            # 全文搜索使用 tsvector 或 LIKE
            # 如果有 tsvector 列，优先使用
            if hasattr(Chunk, "content_tsv"):
                # 使用 PostgreSQL 全文搜索
                stmt = stmt.where(
                    text("content_tsv @@ plainto_tsquery(:query)")
                ).params(query=query)
            else:
                # 使用 LIKE 搜索
                query_pattern = f"%{query}%"
                stmt = stmt.where(Chunk.content.ilike(query_pattern))

            stmt = stmt.order_by(Chunk.created_at.desc()).limit(top_k)

            result = await self.session.execute(stmt)
            chunks = result.scalars().all()

            # 计算分数并转换结果
            results = []
            for chunk in chunks:
                score = self._calculate_score(chunk.content, query)

                if score >= score_threshold:
                    results.append(
                        SearchResult(
                            id=chunk.id,
                            content=chunk.content,
                            score=score,
                            knowledge_id=chunk.knowledge_id,
                            chunk_index=chunk.chunk_index,
                            match_type=SearchResultType.KEYWORD,
                            chunk_type=chunk.chunk_type,
                            parent_chunk_id=chunk.parent_chunk_id,
                            start_at=chunk.start_at,
                            end_at=chunk.end_at,
                        )
                    )

            # 按分数排序
            results.sort(key=lambda x: x.score, reverse=True)

            return results

        except Exception as e:
            logger.warning(
                "keyword_search_failed",
                kb_id=knowledge_base_id,
                error=str(e),
            )
            return []

    def _calculate_score(self, content: str, query: str) -> float:
        """计算关键词匹配分数"""
        content_lower = content.lower()
        query_lower = query.lower()

        # 精确匹配
        if query_lower in content_lower:
            return 1.0

        # 分词匹配
        query_words = query_lower.split()
        if not query_words:
            return 0.0

        match_count = sum(1 for w in query_words if w in content_lower)
        return match_count / len(query_words)


class SearchService:
    """增强搜索服务

    对齐 WeKnora99 + DeerFlow 的混合搜索架构
    """

    def __init__(
        self,
        session: AsyncSession,
        tenant_id: int,
        vector_searcher: VectorSearcher | None = None,
        enable_vector: bool = True,
        enable_keyword: bool = True,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self._vector_searcher = vector_searcher
        self._keyword_searcher = KeywordSearcher(session)
        self._post_processor = SearchResultPostProcessor()
        self.enable_vector = enable_vector
        self.enable_keyword = enable_keyword

    def _get_vector_searcher(self) -> VectorSearcher:
        """获取向量搜索器"""
        if self._vector_searcher is None:
            self._vector_searcher = MemoryVectorSearcher(self.tenant_id)
        return self._vector_searcher

    async def hybrid_search(
        self,
        knowledge_base_id: str,
        query: str,
        top_k: int = 20,
        vector_threshold: float = 0.3,
        keyword_threshold: float = 0.3,
        knowledge_ids: list[str] | None = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> list[dict]:
        """混合搜索

        对齐 WeKnora99 HybridSearch

        Args:
            knowledge_base_id: 知识库 ID
            query: 查询文本
            top_k: 返回结果数量
            vector_threshold: 向量搜索阈值
            keyword_threshold: 关键词搜索阈值
            knowledge_ids: 知识 ID 列表（可选）
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重

        Returns:
            搜索结果列表
        """
        all_results: list[SearchResult] = []
        tasks = []

        # 并行执行向量搜索和关键词搜索
        if self.enable_vector:
            vector_searcher = self._get_vector_searcher()
            tasks.append(
                vector_searcher.search(
                    query=query,
                    knowledge_base_id=knowledge_base_id,
                    top_k=top_k,
                    score_threshold=vector_threshold,
                    knowledge_ids=knowledge_ids,
                )
            )

        if self.enable_keyword:
            tasks.append(
                self._keyword_searcher.search(
                    query=query,
                    knowledge_base_id=knowledge_base_id,
                    tenant_id=self.tenant_id,
                    top_k=top_k,
                    score_threshold=keyword_threshold,
                    knowledge_ids=knowledge_ids,
                )
            )

        if tasks:
            import asyncio

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for results in results_list:
                if isinstance(results, Exception):
                    logger.warning("search_task_failed", error=str(results))
                elif results:
                    all_results.extend(results)

        # 归一化分数并合并
        merged = self._merge_and_normalize(
            all_results,
            vector_weight,
            keyword_weight,
        )

        # 后处理
        processed = await self._post_processor.process(merged)

        # 应用 MMR 增加多样性
        diverse = self._apply_mmr(
            processed[: top_k * 2],  # 取更多候选结果
            min(top_k, len(processed)),
            lambda_param=0.7,
        )

        logger.info(
            "hybrid_search_complete",
            kb_id=knowledge_base_id,
            query=query,
            result_count=len(diverse),
        )

        return [r.to_dict() for r in diverse]

    def _merge_and_normalize(
        self,
        results: list[SearchResult],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        """合并和归一化搜索结果

        对齐 WeKnora99 分数归一化逻辑
        """
        if not results:
            return []

        # 按类型分组
        vector_results = [r for r in results if r.match_type == SearchResultType.VECTOR]
        keyword_results = [r for r in results if r.match_type == SearchResultType.KEYWORD]
        hybrid_results = [r for r in results if r.match_type == SearchResultType.HYBRID]

        # 归一化向量搜索分数
        if vector_results:
            max_vector_score = max(r.score for r in vector_results)
            min_vector_score = min(r.score for r in vector_results)
            vector_range = max_vector_score - min_vector_score or 1

            for r in vector_results:
                r.score = (r.score - min_vector_score) / vector_range * vector_weight

        # 归一化关键词搜索分数
        if keyword_results:
            max_keyword_score = max(r.score for r in keyword_results)
            min_keyword_score = min(r.score for r in keyword_results)
            keyword_range = max_keyword_score - min_keyword_score or 1

            for r in keyword_results:
                r.score = (r.score - min_keyword_score) / keyword_range * keyword_weight

        # 混合结果保持原分数
        # Hybrid results already have normalized scores

        # 合并并排序
        merged = vector_results + keyword_results + hybrid_results
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged

    def _apply_mmr(
        self,
        results: list[SearchResult],
        k: int,
        lambda_param: float = 0.7,
    ) -> list[SearchResult]:
        """应用 MMR 算法增加多样性

        对齐 WeKnora99 applyMMR
        """
        if k <= 0 or not results:
            return []

        selected: list[SearchResult] = []
        selected_indices: set[int] = set()

        # 预计算 token sets
        token_sets = [self._tokenize(r.content) for r in results]

        while len(selected) < k and len(selected_indices) < len(results):
            best_idx = -1
            best_score = -1.0

            for i, r in enumerate(results):
                if i in selected_indices:
                    continue

                relevance = r.score
                redundancy = 0.0

                # 计算与已选结果的最大相似度
                for sel_idx in selected_indices:
                    sim = self._jaccard(token_sets[i], token_sets[sel_idx])
                    redundancy = max(redundancy, sim)

                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx < 0:
                break

            selected.append(results[best_idx])
            selected_indices.add(best_idx)

        return selected

    def _tokenize(self, text: str) -> set[str]:
        """简单的分词"""
        words = re.findall(r"\w+", text.lower())
        return set(words)

    def _jaccard(self, set1: set[str], set2: set[str]) -> float:
        """计算 Jaccard 相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


# 便捷函数
async def hybrid_search(
    session: AsyncSession,
    tenant_id: int,
    knowledge_base_id: str,
    query: str,
    top_k: int = 20,
    **kwargs,
) -> list[dict]:
    """混合搜索便捷函数"""
    service = SearchService(session, tenant_id)
    return await service.hybrid_search(
        knowledge_base_id=knowledge_base_id,
        query=query,
        top_k=top_k,
        **kwargs,
    )


__all__ = [
    "SearchService",
    "SearchResult",
    "SearchResultType",
    "SearchResultPostProcessor",
    "VectorSearcher",
    "QdrantVectorSearcher",
    "MemoryVectorSearcher",
    "KeywordSearcher",
    "hybrid_search",
]
