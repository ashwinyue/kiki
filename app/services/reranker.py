"""重排序服务

对齐 WeKnora99 的重排序功能，支持：
- Cohere Rerank API
- 本地重排序模型（可选）
- 多种重排序策略

参考:
- WeKnora99 重排序相关实现
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 重排序结果 ==============


@dataclass
class RerankResult:
    """重排序结果"""

    chunk_id: str
    score: float
    index: int  # 重排序后的位置


# ============== 重排序器接口 ==============


class Reranker(ABC):
    """重排序器接口"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],  # [(doc_id, content), ...]
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 文档列表 [(doc_id, content), ...]
            top_n: 返回前 N 个结果

        Returns:
            重排序结果列表
        """
        pass


# ============== Cohere 重排序器 ==============


class CohereReranker(Reranker):
    """Cohere Rerank API 重排序器

    使用 Cohere 的 Rerank API 进行重排序
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v2",
        base_url: str = "https://api.cohere.ai/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            logger.warning("cohere_api_key_not_set")

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """使用 Cohere Rerank API 重排序文档

        Args:
            query: 查询文本
            documents: 文档列表 [(doc_id, content), ...]
            top_n: 返回前 N 个结果

        Returns:
            重排序结果列表
        """
        if not self.api_key:
            logger.warning("cohere_rerank_skipped_no_api_key")
            # 返回原始顺序
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        logger.info(
            "cohere_rerank_start",
            query=query,
            doc_count=len(documents),
            model=self.model,
        )

        # 准备请求数据
        docs = [content for _, content in documents]
        doc_ids = [doc_id for doc_id, _ in documents]

        request_data = {
            "query": query,
            "documents": docs,
            "model": self.model,
            "top_n": top_n or len(documents),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Client-Name": "kiki",
                    },
                    json=request_data,
                )

                response.raise_for_status()
                result_data = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "cohere_rerank_http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            # 返回原始顺序
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        except Exception as e:
            logger.error(
                "cohere_rerank_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            # 返回原始顺序
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        # 解析结果
        results = []
        for item in result_data.get("results", []):
            index = item.get("index")
            relevance_score = item.get("relevance_score", 0.0)

            if index is not None and 0 <= index < len(doc_ids):
                results.append(
                    RerankResult(
                        chunk_id=doc_ids[index],
                        score=relevance_score,
                        index=len(results),
                    )
                )

        logger.info(
            "cohere_rerank_complete",
            result_count=len(results),
        )

        return results


# ============== Jina Reranker ==============


class JinaReranker(Reranker):
    """Jina Reranker API 重排序器

    使用 Jina 的 Reranker API 进行重排序
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
        base_url: str = "https://api.jina.ai/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            logger.warning("jina_api_key_not_set")

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """使用 Jina Reranker API 重排序文档

        Args:
            query: 查询文本
            documents: 文档列表 [(doc_id, content), ...]
            top_n: 返回前 N 个结果

        Returns:
            重排序结果列表
        """
        if not self.api_key:
            logger.warning("jina_rerank_skipped_no_api_key")
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        logger.info(
            "jina_rerank_start",
            query=query,
            doc_count=len(documents),
            model=self.model,
        )

        # 准备请求数据
        docs = [content for _, content in documents]
        doc_ids = [doc_id for doc_id, _ in documents]

        request_data = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": top_n or len(documents),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_data,
                )

                response.raise_for_status()
                result_data = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "jina_rerank_http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        except Exception as e:
            logger.error(
                "jina_rerank_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return [
                RerankResult(chunk_id=doc_id, score=1.0 - i * 0.01, index=i)
                for i, (doc_id, _) in enumerate(documents)
            ]

        # 解析结果
        results = []
        for item in result_data.get("results", []):
            index = item.get("index")
            relevance_score = item.get("relevance_score", 0.0)

            if index is not None and 0 <= index < len(doc_ids):
                results.append(
                    RerankResult(
                        chunk_id=doc_ids[index],
                        score=relevance_score,
                        index=len(results),
                    )
                )

        logger.info(
            "jina_rerank_complete",
            result_count=len(results),
        )

        return results


# ============== 本地重排序器（基于相似度）============


class LocalReranker(Reranker):
    """本地重排序器

    使用简单的词汇相似度进行重排序
    适用于没有 API Key 的情况
    """

    def __init__(self) -> None:
        pass

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """使用本地相似度算法重排序文档

        Args:
            query: 查询文本
            documents: 文档列表 [(doc_id, content), ...]
            top_n: 返回前 N 个结果

        Returns:
            重排序结果列表
        """
        logger.info(
            "local_rerank_start",
            query=query,
            doc_count=len(documents),
        )

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # 计算每个文档的相似度分数
        scored_docs = []
        for doc_id, content in documents:
            score = self._compute_similarity(query_lower, query_words, content)
            scored_docs.append((doc_id, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 构建结果
        results = [
            RerankResult(chunk_id=doc_id, score=score, index=i)
            for i, (doc_id, score) in enumerate(scored_docs[: top_n or len(scored_docs)])
        ]

        logger.info(
            "local_rerank_complete",
            result_count=len(results),
        )

        return results

    def _compute_similarity(
        self,
        query_lower: str,
        query_words: set[str],
        content: str,
    ) -> float:
        """计算相似度分数

        Args:
            query_lower: 小写的查询文本
            query_words: 查询词集合
            content: 文档内容

        Returns:
            相似度分数 (0-1)
        """
        content_lower = content.lower()

        # 精确匹配加分
        exact_match_bonus = 0.0
        if query_lower in content_lower:
            exact_match_bonus = 0.3
            count = content_lower.count(query_lower)
            exact_match_bonus += min(0.2, count * 0.05)

        # 词汇重叠分数
        content_words = set(content_lower.split())
        if not query_words:
            word_overlap = 0.0
        else:
            intersection = query_words & content_words
            word_overlap = len(intersection) / len(query_words)

        return exact_match_bonus + word_overlap * 0.7


# ============== 重排序服务 ==============


class RerankerService:
    """重排序服务

    管理多种重排序器，根据模型 ID 选择合适的重排序器
    """

    def __init__(self) -> None:
        self._rerankers: dict[str, Reranker] = {}

    def _get_reranker(self, model_id: str) -> Reranker:
        """获取重排序器

        Args:
            model_id: 模型 ID

        Returns:
            重排序器实例
        """
        if model_id in self._rerankers:
            return self._rerankers[model_id]

        # 根据 model_id 选择重排序器
        if model_id.startswith("cohere"):
            reranker = CohereReranker(model=model_id)
        elif model_id.startswith("jina"):
            reranker = JinaReranker(model=model_id)
        else:
            # 默认使用本地重排序器
            reranker = LocalReranker()

        self._rerankers[model_id] = reranker
        return reranker

    async def rerank(
        self,
        query: str,
        chunk_ids: list[str],
        model_id: str,
        session: AsyncSession,
        top_n: int | None = None,
    ) -> list[str]:
        """重排序分块 ID 列表

        Args:
            query: 查询文本
            chunk_ids: 分块 ID 列表
            model_id: 重排序模型 ID
            session: 数据库会话
            top_n: 返回前 N 个结果

        Returns:
            重排序后的分块 ID 列表
        """
        if not chunk_ids:
            return []

        logger.info(
            "rerank_start",
            query=query,
            chunk_count=len(chunk_ids),
            model_id=model_id,
        )

        # 获取分块内容
        from sqlalchemy import select

        from app.models.knowledge import Chunk

        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        # 构建文档列表
        chunk_map = {c.id: c for c in chunks}
        documents = [
            (chunk_id, chunk_map[chunk_id].content)
            for chunk_id in chunk_ids
            if chunk_id in chunk_map
        ]

        if not documents:
            return []

        # 获取重排序器并执行重排序
        reranker = self._get_reranker(model_id)
        rerank_results = await reranker.rerank(query, documents, top_n)

        # 返回重排序后的 ID 列表
        reranked_ids = [r.chunk_id for r in rerank_results]

        logger.info(
            "rerank_complete",
            result_count=len(reranked_ids),
        )

        return reranked_ids


__all__ = [
    "RerankResult",
    "Reranker",
    "CohereReranker",
    "JinaReranker",
    "LocalReranker",
    "RerankerService",
]
