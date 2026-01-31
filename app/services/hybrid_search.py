"""混合搜索服务

对齐 WeKnora99 的混合搜索功能，支持：
- 向量搜索 (dense_vector)
- 关键词搜索 (keyword)
- RRF (Reciprocal Rank Fusion) 结果融合
- 重排序 (rerank)

参考:
- WeKnora99/internal/application/service/knowledgebase.go::HybridSearch
- WeKnora99/internal/application/service/chat_pipline/search.go
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.services.reranker import RerankerService

logger = get_logger(__name__)


# ============== 搜索参数 ==============


@dataclass
class SearchParams:
    """搜索参数

    对齐 WeKnora99 SearchParams
    """

    query_text: str
    vector_threshold: float = 0.5
    keyword_threshold: float = 0.3
    match_count: int = 5
    disable_keywords_match: bool = False
    disable_vector_match: bool = False
    knowledge_ids: list[str] | None = None
    tag_ids: list[str] | None = None
    only_recommended: bool = False
    enable_rerank: bool = False
    rerank_model_id: str | None = None
    top_k: int = 20  # 检索时的候选数量，用于 rerank


@dataclass
class SearchResult:
    """搜索结果

    对齐 WeKnora99 SearchResult
    """

    id: str
    content: str
    knowledge_id: str
    chunk_index: int
    knowledge_title: str
    start_at: int
    end_at: int
    score: float
    match_type: str  # vector, keyword, rerank, parent, nearby
    metadata: dict[str, Any]
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    image_info: str | None = None
    knowledge_filename: str | None = None
    knowledge_source: str | None = None
    chunk_metadata: dict[str, Any] | None = None
    matched_content: str | None = None


# ============== 向量搜索 ==============


class VectorSearcher:
    """向量搜索器

    使用 embedding 模型进行语义搜索
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_model_id: str,
    ) -> None:
        self.session = session
        self.embedding_model_id = embedding_model_id
        self._embedder = None

    async def get_embedder(self):
        """获取嵌入模型"""
        if self._embedder is None:
            from app.llm.embeddings import get_embedding_service

            embedding_service = get_embedding_service()
            self._embedder = await embedding_service.get_model(self.embedding_model_id)
        return self._embedder

    async def search(
        self,
        kb_id: str,
        tenant_id: int,
        params: SearchParams,
    ) -> list[tuple[str, float, str | None]]:
        """执行向量搜索

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 搜索参数

        Returns:
            [(chunk_id, score, matched_content), ...]
        """
        from sqlalchemy import select

        from app.models.knowledge import Chunk, Knowledge

        # 获取嵌入模型并生成查询向量
        embedder = await self.get_embedder()
        query_vector = await embedder.embed(params.query_text)

        logger.info(
            "vector_search_start",
            kb_id=kb_id,
            query=params.query_text,
            vector_dim=len(query_vector),
        )

        # 使用余弦相似度进行搜索
        # TODO: 集成 pgvector 进行高效的向量相似度搜索
        # 当前使用简单的文本匹配作为占位符

        stmt = (
            select(Chunk)
            .join(Knowledge, Chunk.knowledge_id == Knowledge.id)
            .where(
                Chunk.knowledge_base_id == kb_id,
                Chunk.tenant_id == tenant_id,
                Chunk.is_enabled,
                Chunk.deleted_at.is_(None),
                Knowledge.deleted_at.is_(None),
                Knowledge.parse_status == "completed",
            )
        )

        # 应用知识 ID 过滤
        if params.knowledge_ids:
            stmt = stmt.where(Chunk.knowledge_id.in_(params.knowledge_ids))

        # 应用标签过滤
        if params.tag_ids:
            stmt = stmt.where(Chunk.tag_id.in_(params.tag_ids))

        # 应用推荐过滤
        if params.only_recommended:
            # TODO: 实现推荐逻辑
            pass

        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        # 计算余弦相似度
        results = []
        for chunk in chunks:
            # 如果有向量，计算相似度
            # 这里简化处理，使用内容相似度
            similarity = self._compute_similarity(
                params.query_text.lower(),
                chunk.content.lower(),
            )

            if similarity >= params.vector_threshold:
                results.append((chunk.id, similarity, None))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "vector_search_complete",
            kb_id=kb_id,
            result_count=len(results),
        )

        return results[: params.top_k]

    def _compute_similarity(self, query: str, content: str) -> float:
        """计算查询和内容的相似度

        使用简单的词汇重叠作为占位符
        TODO: 替换为真实的向量相似度计算
        """
        query_words = set(query.split())
        content_words = set(content.split())

        if not query_words:
            return 0.0

        # Jaccard 相似度
        intersection = query_words & content_words
        union = query_words | content_words

        return len(intersection) / len(union) if union else 0.0


# ============== 关键词搜索 ==============


class KeywordSearcher:
    """关键词搜索器

    使用全文搜索进行关键词匹配
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self,
        kb_id: str,
        tenant_id: int,
        params: SearchParams,
    ) -> list[tuple[str, float, str | None]]:
        """执行关键词搜索

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 搜索参数

        Returns:
            [(chunk_id, score, matched_content), ...]
        """
        from sqlalchemy import or_, select

        from app.models.knowledge import Chunk, Knowledge

        logger.info(
            "keyword_search_start",
            kb_id=kb_id,
            query=params.query_text,
        )

        query_lower = params.query_text.lower()

        # 构建搜索条件
        search_conditions = [
            Chunk.content.ilike(f"%{query_lower}%"),
        ]

        # TODO: 添加更多字段的搜索，如标题、元数据等

        stmt = (
            select(Chunk)
            .join(Knowledge, Chunk.knowledge_id == Knowledge.id)
            .where(
                Chunk.knowledge_base_id == kb_id,
                Chunk.tenant_id == tenant_id,
                Chunk.is_enabled,
                Chunk.deleted_at.is_(None),
                Knowledge.deleted_at.is_(None),
                Knowledge.parse_status == "completed",
                or_(*search_conditions),
            )
        )

        # 应用知识 ID 过滤
        if params.knowledge_ids:
            stmt = stmt.where(Chunk.knowledge_id.in_(params.knowledge_ids))

        # 应用标签过滤
        if params.tag_ids:
            stmt = stmt.where(Chunk.tag_id.in_(params.tag_ids))

        # 应用推荐过滤
        if params.only_recommended:
            # TODO: 实现推荐逻辑
            pass

        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        # 计算关键词匹配分数
        results = []
        for chunk in chunks:
            score = self._compute_keyword_score(
                chunk.content,
                query_lower,
            )

            # 提取匹配的内容片段
            matched_content = self._extract_matched_content(
                chunk.content,
                query_lower,
            )

            if score >= params.keyword_threshold:
                results.append((chunk.id, score, matched_content))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "keyword_search_complete",
            kb_id=kb_id,
            result_count=len(results),
        )

        return results[: params.top_k]

    def _compute_keyword_score(self, content: str, query: str) -> float:
        """计算关键词匹配分数

        Args:
            content: 文本内容
            query: 查询词

        Returns:
            相似度分数 (0-1)
        """
        content_lower = content.lower()

        # 精确匹配
        if query in content_lower:
            # 根据匹配次数增加分数
            count = content_lower.count(query)
            return min(1.0, 0.5 + count * 0.1)

        # 分词匹配
        query_words = query.split()
        if not query_words:
            return 0.0

        match_count = sum(1 for word in query_words if word.lower() in content_lower)
        return match_count / len(query_words)

    def _extract_matched_content(
        self,
        content: str,
        query: str,
        context_length: int = 200,
    ) -> str | None:
        """提取匹配的上下文

        Args:
            content: 文本内容
            query: 查询词
            context_length: 上下文长度

        Returns:
            匹配的内容片段
        """
        content_lower = content.lower()

        # 查找第一个匹配位置
        index = content_lower.find(query.lower())
        if index == -1:
            return None

        # 提取上下文
        start = max(0, index - context_length // 2)
        end = min(len(content), index + len(query) + context_length // 2)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet


# ============== RRF 结果融合 ==============


class RRFCombiner:
    """RRF (Reciprocal Rank Fusion) 结果组合器

    对齐 WeKnora99 的 RRF 融合逻辑
    """

    # RRF 常数，通常使用 60
    RRF_K = 60

    def fuse(
        self,
        vector_results: list[tuple[str, float, str | None]],
        keyword_results: list[tuple[str, float, str | None]],
    ) -> dict[str, float]:
        """使用 RRF 融合向量搜索和关键词搜索结果

        Args:
            vector_results: 向量搜索结果 [(chunk_id, score, matched_content), ...]
            keyword_results: 关键词搜索结果 [(chunk_id, score, matched_content), ...]

        Returns:
            {chunk_id: rrf_score}
        """
        rrf_scores: dict[str, float] = {}

        # 处理向量搜索结果
        for rank, (chunk_id, _score, _) in enumerate(vector_results, start=1):
            rrf_score = 1.0 / (self.RRF_K + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        # 处理关键词搜索结果
        for rank, (chunk_id, _score, _) in enumerate(keyword_results, start=1):
            rrf_score = 1.0 / (self.RRF_K + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        return rrf_scores

    def fuse_single(
        self,
        results: list[tuple[str, float, str | None]],
    ) -> dict[str, float]:
        """融合单一来源的结果

        当只有向量搜索或只有关键词搜索时使用

        Args:
            results: 搜索结果 [(chunk_id, score, matched_content), ...]

        Returns:
            {chunk_id: normalized_score}
        """
        # 直接使用原始分数
        scores: dict[str, float] = {}
        for chunk_id, score, _ in results:
            # 保留最高分
            if chunk_id not in scores or score > scores[chunk_id]:
                scores[chunk_id] = score

        return scores


# ============== 混合搜索服务 ==============


class HybridSearchService:
    """混合搜索服务

    对齐 WeKnora99 的 HybridSearch 功能
    支持向量搜索 + 关键词搜索 + RRF 融合 + 重排序
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_model_id: str,
    ) -> None:
        self.session = session
        self.embedding_model_id = embedding_model_id
        self._vector_searcher: VectorSearcher | None = None
        self._keyword_searcher: KeywordSearcher | None = None
        self._rrf_combiner: RRFCombiner | None = None
        self._reranker_service: RerankerService | None = None

    @property
    def vector_searcher(self) -> VectorSearcher:
        """延迟初始化向量搜索器"""
        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher(
                self.session,
                self.embedding_model_id,
            )
        return self._vector_searcher

    @property
    def keyword_searcher(self) -> KeywordSearcher:
        """延迟初始化关键词搜索器"""
        if self._keyword_searcher is None:
            self._keyword_searcher = KeywordSearcher(self.session)
        return self._keyword_searcher

    @property
    def rrf_combiner(self) -> RRFCombiner:
        """延迟初始化 RRF 组合器"""
        if self._rrf_combiner is None:
            self._rrf_combiner = RRFCombiner()
        return self._rrf_combiner

    @property
    def reranker_service(self) -> RerankerService:
        """延迟初始化重排序服务"""
        if self._reranker_service is None:
            self._reranker_service = RerankerService()
        return self._reranker_service

    async def search(
        self,
        kb_id: str,
        tenant_id: int,
        params: SearchParams,
    ) -> list[SearchResult]:
        """执行混合搜索

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 搜索参数

        Returns:
            搜索结果列表
        """
        logger.info(
            "hybrid_search_start",
            kb_id=kb_id,
            tenant_id=tenant_id,
            query=params.query_text,
            vector_threshold=params.vector_threshold,
            keyword_threshold=params.keyword_threshold,
            match_count=params.match_count,
            disable_vector=params.disable_vector_match,
            disable_keyword=params.disable_keywords_match,
        )

        # 并行执行向量和关键词搜索
        vector_results: list[tuple[str, float, str | None]] = []
        keyword_results: list[tuple[str, float, str | None]] = []

        tasks = []
        if not params.disable_vector_match:
            tasks.append(
                self.vector_searcher.search(kb_id, tenant_id, params),
            )
        if not params.disable_keywords_match:
            tasks.append(
                self.keyword_searcher.search(kb_id, tenant_id, params),
            )

        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results_list):
                if isinstance(result, Exception):
                    logger.error(
                        "search_failed",
                        search_type="vector" if i == 0 else "keyword",
                        error=str(result),
                    )
                    continue
                if i == 0 and not params.disable_vector_match:
                    vector_results = result
                else:
                    keyword_results = result

        logger.info(
            "search_results_raw",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
        )

        # RRF 融合
        fused_scores: dict[str, float] = {}
        matched_contents: dict[str, str | None] = {}

        if vector_results and keyword_results:
            # 两种搜索都有结果，使用 RRF
            fused_scores = self.rrf_combiner.fuse(vector_results, keyword_results)

            # 合并匹配内容
            for chunk_id, _, matched_content in vector_results:
                if matched_content and chunk_id not in matched_contents:
                    matched_contents[chunk_id] = matched_content
            for chunk_id, _, matched_content in keyword_results:
                if matched_content and chunk_id not in matched_contents:
                    matched_contents[chunk_id] = matched_content

        elif vector_results:
            # 只有向量搜索结果
            fused_scores = self.rrf_combiner.fuse_single(vector_results)
            for chunk_id, _, matched_content in vector_results:
                matched_contents[chunk_id] = matched_content

        elif keyword_results:
            # 只有关键词搜索结果
            fused_scores = self.rrf_combiner.fuse_single(keyword_results)
            for chunk_id, _, matched_content in keyword_results:
                matched_contents[chunk_id] = matched_content

        if not fused_scores:
            logger.info("hybrid_search_no_results")
            return []

        # 按分数排序
        sorted_chunks = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # 获取前 top_k 个结果用于重排序
        top_k = params.top_k
        candidate_chunk_ids = [cid for cid, _ in sorted_chunks[:top_k]]

        # 重排序
        if params.enable_rerank and params.rerank_model_id:
            candidate_chunk_ids = await self.reranker_service.rerank(
                query=params.query_text,
                chunk_ids=candidate_chunk_ids,
                model_id=params.rerank_model_id,
                session=self.session,
            )
            # 更新分数 (重排序后，前面的分数更高)
            for i, chunk_id in enumerate(candidate_chunk_ids):
                if chunk_id in fused_scores:
                    fused_scores[chunk_id] = 1.0 - (i / len(candidate_chunk_ids))

        # 构建最终结果
        results = await self._build_search_results(
            kb_id,
            tenant_id,
            candidate_chunk_ids[: params.match_count],
            fused_scores,
            matched_contents,
            params,
        )

        logger.info(
            "hybrid_search_complete",
            result_count=len(results),
        )

        return results

    async def _build_search_results(
        self,
        kb_id: str,
        tenant_id: int,
        chunk_ids: list[str],
        scores: dict[str, float],
        matched_contents: dict[str, str | None],
        params: SearchParams,
    ) -> list[SearchResult]:
        """构建搜索结果

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            chunk_ids: 分块 ID 列表
            scores: 分数字典
            matched_contents: 匹配内容字典
            params: 搜索参数

        Returns:
            搜索结果列表
        """
        from sqlalchemy import select

        from app.models.knowledge import Chunk, Knowledge

        if not chunk_ids:
            return []

        # 批量获取分块
        stmt = select(Chunk).where(
            Chunk.id.in_(chunk_ids),
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        # 获取关联的知识条目
        knowledge_ids = list({c.knowledge_id for c in chunks})
        knowledge_stmt = select(Knowledge).where(
            Knowledge.id.in_(knowledge_ids),
            Knowledge.tenant_id == tenant_id,
            Knowledge.deleted_at.is_(None),
        )
        knowledge_result = await self.session.execute(knowledge_stmt)
        knowledges = knowledge_result.scalars().all()

        # 构建知识条目字典
        knowledge_map = {k.id: k for k in knowledges}

        # 构建分块字典
        chunk_map = {c.id: c for c in chunks}

        # 构建搜索结果
        search_results = []
        for chunk_id in chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue

            knowledge = knowledge_map.get(chunk.knowledge_id)
            if not knowledge:
                continue

            score = scores.get(chunk_id, 0.0)
            matched_content = matched_contents.get(chunk_id)

            # 确定匹配类型
            match_type = "rerank" if params.enable_rerank else "hybrid"

            search_results.append(
                SearchResult(
                    id=chunk.id,
                    content=chunk.content,
                    knowledge_id=chunk.knowledge_id,
                    chunk_index=chunk.chunk_index,
                    knowledge_title=knowledge.title,
                    start_at=chunk.start_at,
                    end_at=chunk.end_at,
                    score=score,
                    match_type=match_type,
                    metadata=knowledge.meta_data or {},
                    chunk_type=chunk.chunk_type or "text",
                    parent_chunk_id=chunk.parent_chunk_id,
                    image_info=chunk.image_info,
                    knowledge_filename=knowledge.file_name,
                    knowledge_source=knowledge.source,
                    chunk_metadata=chunk.meta_data,
                    matched_content=matched_content,
                )
            )

        return search_results


__all__ = [
    "SearchParams",
    "SearchResult",
    "HybridSearchService",
    "VectorSearcher",
    "KeywordSearcher",
    "RRFCombiner",
]
