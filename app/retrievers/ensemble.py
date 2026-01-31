"""集成检索器

融合多个检索器结果，支持多种融合策略（RRF、加权平均等）。

使用示例:
```python
from app.retrievers import EnsembleRetriever, EnsembleRetrieverConfig
from app.retrievers import BM25Retriever, BM25RetrieverConfig
from app.vector_stores import create_vector_store, VectorStoreConfig

# 创建向量检索器
vector_store = create_vector_store("memory", VectorStoreConfig())
vector_retriever = await vector_store.as_retriever(k=5)

# 创建 BM25 检索器
bm25_config = BM25RetrieverConfig(k=5)
bm25_retriever = BM25Retriever(bm25_config)
await bm25_retriever.from_texts(["文档1", "文档2"])

# 创建集成检索器
ensemble_config = EnsembleRetrieverConfig(
    weights=[0.7, 0.3],  # 向量 70%, BM25 30%
)
ensemble = EnsembleRetriever(
    config=ensemble_config,
    retrievers=[vector_retriever, bm25_retriever],
)
results = await ensemble.aretrieve("查询内容")
```
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleRetrieverConfig:
    """集成检索器配置

    Attributes:
        weights: 各检索器权重列表
        c: RRF 常数（仅用于 RRF 融合）
        fusion_strategy: 融合策略 (rrf/weighted_sum/relative_score)
        tenant_id: 租户 ID
    """

    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    c: int = 60  # RRF 常数
    fusion_strategy: Literal["rrf", "weighted_sum", "relative_score"] = "rrf"
    tenant_id: int | None = None


class EnsembleRetriever(BaseRetriever):
    """集成检索器

    融合多个检索器的结果，提高检索召回率和准确率。
    """

    config: EnsembleRetrieverConfig
    _retrievers: list[BaseRetriever]

    def __init__(
        self,
        config: EnsembleRetrieverConfig,
        retrievers: list[BaseRetriever],
    ):
        """初始化集成检索器

        Args:
            config: 检索器配置
            retrievers: 检索器列表
        """
        super().__init__()
        self.config = config
        self._retrievers = retrievers

        # 验证权重数量
        if len(config.weights) != len(retrievers):
            logger.warning(
                "ensemble_weights_mismatch",
                weights_count=len(config.weights),
                retrievers_count=len(retrievers),
                message="将使用均匀权重",
            )
            # 使用均匀权重
            weight = 1.0 / len(retrievers)
            config.weights = [weight] * len(retrievers)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> list[Document]:
        """同步检索文档（LangChain 接口）

        Args:
            query: 查询文本
            run_manager: 运行管理器
            **kwargs: 额外参数

        Returns:
            文档列表
        """
        import asyncio

        return asyncio.run(self.aretrieve(query=query))

    async def aretrieve(
        self,
        query: str,
    ) -> list[Document]:
        """异步检索文档

        Args:
            query: 查询文本

        Returns:
            融合后的文档列表
        """
        if not self._retrievers:
            logger.warning("ensemble_no_retrievers")
            return []

        # 执行所有检索器
        all_results = []
        for i, retriever in enumerate(self._retrievers):
            try:
                if hasattr(retriever, "aretrieve"):
                    docs = await retriever.aretrieve(query)
                else:
                    docs = retriever.get_relevant_documents(query)

                # 添加权重到元数据
                weight = self.config.weights[i] if i < len(self.config.weights) else 0.0
                for doc in docs:
                    doc.metadata["_retriever_weight"] = weight
                    doc.metadata["_retriever_index"] = i

                all_results.append(docs)

                logger.info(
                    "ensemble_retriever_completed",
                    retriever_index=i,
                    result_count=len(docs),
                )

            except Exception as e:
                logger.error(
                    "ensemble_retriever_failed",
                    retriever_index=i,
                    error=str(e),
                )

        # 融合结果
        fused_results = self._fuse_results(all_results, query)

        logger.info(
            "ensemble_retrieve_completed",
            query=query[:100],
            result_count=len(fused_results),
            retrievers_count=len(self._retrievers),
        )

        return fused_results

    def _fuse_results(
        self,
        all_results: list[list[Document]],
        query: str,
    ) -> list[Document]:
        """融合多个检索器的结果

        Args:
            all_results: 所有检索器结果列表
            query: 查询文本

        Returns:
            融合后的文档列表
        """
        strategy = self.config.fusion_strategy

        if strategy == "rrf":
            return self._rrf_fusion(all_results)
        elif strategy == "weighted_sum":
            return self._weighted_sum_fusion(all_results)
        else:  # relative_score
            return self._relative_score_fusion(all_results)

    def _rrf_fusion(
        self,
        all_results: list[list[Document]],
    ) -> list[Document]:
        """Reciprocal Rank Fusion 融合

        RRF 公式: score = sum(weight / (c + rank))

        Args:
            all_results: 所有检索器结果列表

        Returns:
            融合后的文档列表
        """
        # 使用文档内容作为唯一标识
        doc_scores: dict[str, tuple[float, Document]] = {}

        c = self.config.c

        for retriever_idx, docs in enumerate(all_results):
            weight = (
                self.config.weights[retriever_idx]
                if retriever_idx < len(self.config.weights)
                else 1.0
            )

            for rank, doc in enumerate(docs, start=1):
                # 使用内容作为键（简化实现）
                key = doc.page_content[:200]  # 使用前200字符作为唯一标识

                if key not in doc_scores:
                    doc_scores[key] = (0.0, doc)

                current_score, _ = doc_scores[key]
                rrf_score = weight / (c + rank)
                doc_scores[key] = (current_score + rrf_score, doc)

        # 排序并返回
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )

        # 更新分数到元数据
        results = []
        for score, doc in sorted_docs:
            doc.metadata["_fusion_score"] = score
            doc.metadata["_fusion_method"] = "rrf"
            results.append(doc)

        return results

    def _weighted_sum_fusion(
        self,
        all_results: list[list[Document]],
    ) -> list[Document]:
        """加权平均融合

        假设文档有 score 元数据。

        Args:
            all_results: 所有检索器结果列表

        Returns:
            融合后的文档列表
        """
        doc_scores: dict[str, tuple[float, Document]] = {}

        for retriever_idx, docs in enumerate(all_results):
            weight = (
                self.config.weights[retriever_idx]
                if retriever_idx < len(self.config.weights)
                else 1.0
            )

            for doc in docs:
                key = doc.page_content[:200]

                # 获取原始分数
                original_score = doc.metadata.get("score", 1.0)
                weighted_score = original_score * weight

                if key not in doc_scores:
                    doc_scores[key] = (0.0, doc)

                current_score, _ = doc_scores[key]
                doc_scores[key] = (current_score + weighted_score, doc)

        # 排序并返回
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in sorted_docs:
            doc.metadata["_fusion_score"] = score
            doc.metadata["_fusion_method"] = "weighted_sum"
            results.append(doc)

        return results

    def _relative_score_fusion(
        self,
        all_results: list[list[Document]],
    ) -> list[Document]:
        """相对分数融合

        将分数归一化到 0-1 范围后融合。

        Args:
            all_results: 所有检索器结果列表

        Returns:
            融合后的文档列表
        """
        doc_scores: dict[str, tuple[float, Document]] = {}

        for retriever_idx, docs in enumerate(all_results):
            weight = (
                self.config.weights[retriever_idx]
                if retriever_idx < len(self.config.weights)
                else 1.0
            )

            if not docs:
                continue

            # 获取该检索器的分数范围
            scores = [d.metadata.get("score", 0) for d in docs]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

            for doc in docs:
                key = doc.page_content[:200]

                # 归一化分数
                original_score = doc.metadata.get("score", 0)
                normalized_score = (original_score - min_score) / score_range
                weighted_score = normalized_score * weight

                if key not in doc_scores:
                    doc_scores[key] = (0.0, doc)

                current_score, _ = doc_scores[key]
                doc_scores[key] = (current_score + weighted_score, doc)

        # 排序并返回
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in sorted_docs:
            doc.metadata["_fusion_score"] = score
            doc.metadata["_fusion_method"] = "relative_score"
            results.append(doc)

        return results

    def add_retriever(
        self,
        retriever: BaseRetriever,
        weight: float = 0.5,
    ) -> None:
        """添加检索器

        Args:
            retriever: 检索器实例
            weight: 权重
        """
        self._retrievers.append(retriever)
        self.config.weights.append(weight)

        logger.info(
            "ensemble_retriever_added",
            total_retrievers=len(self._retrievers),
        )

    def remove_retriever(self, index: int) -> None:
        """移除检索器

        Args:
            index: 检索器索引
        """
        if 0 <= index < len(self._retrievers):
            self._retrievers.pop(index)
            self.config.weights.pop(index)

            logger.info(
                "ensemble_retriever_removed",
                remaining_retrievers=len(self._retrievers),
            )


__all__ = [
    "EnsembleRetriever",
    "EnsembleRetrieverConfig",
]
