"""评估指标基类

提供评估指标的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.types import RetrievalMetrics, GenerationMetrics


class BaseMetric(ABC, BaseModel):
    """评估指标基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称"""
        pass

    @abstractmethod
    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        """计算检索指标

        Args:
            retrieved_ids: 检索结果 ID 列表
            relevant_ids: 正确答案 ID 列表

        Returns:
            指标分数
        """
        pass

    @abstractmethod
    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        """计算生成指标

        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表

        Returns:
            指标分数
        """
        pass


class MetricInput(BaseModel):
    """指标计算输入"""

    retrieved_ids: list[list[str]] = Field(default_factory=list)
    relevant_ids: list[list[str]] = Field(default_factory=list)
    generated_texts: list[str] = Field(default_factory=list)
    reference_texts: list[str] = Field(default_factory=list)


class PrecisionMetric(BaseMetric):
    """Precision@K 指标"""

    k: int = 10

    @property
    def name(self) -> str:
        return f"precision@{self.k}"

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        if not retrieved_ids:
            return 0.0

        total_precision = 0.0
        for i, (retrieved, relevant) in enumerate(zip(retrieved_ids, relevant_ids)):
            if i >= self.k:
                break

            hits = len(set(retrieved[: self.k]) & set(relevant))
            precision = hits / self.k if self.k > 0 else 0.0
            total_precision += precision

        return total_precision / len(retrieved_ids[: self.k])

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        # Precision 不适用于生成任务
        return 0.0


class RecallMetric(BaseMetric):
    """Recall@K 指标"""

    k: int = 10

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        if not relevant_ids:
            return 0.0

        total_recall = 0.0
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            hits = len(set(retrieved[: self.k]) & set(relevant))
            relevant_count = len(set(relevant))
            recall = hits / relevant_count if relevant_count > 0 else 0.0
            total_recall += recall

        return total_recall / len(retrieved_ids)

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        return 0.0


class NDCGMetric(BaseMetric):
    """NDCG (Normalized Discounted Cumulative Gain) 指标"""

    k: int = 10

    @property
    def name(self) -> str:
        return f"ndcg@{self.k}"

    def _dcg(self, retrieved: list[str], relevant: list[str]) -> float:
        """计算 DCG"""
        dcg = 0.0
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved[: self.k]):
            if doc_id in relevant_set:
                dcg += 1.0 / (i + 1)
        return dcg

    def _idcg(self, relevant: list[str]) -> float:
        """计算 IDCG"""
        return sum(1.0 / (i + 1) for i in range(min(len(relevant), self.k)))

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        if not retrieved_ids:
            return 0.0

        total_ndcg = 0.0
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            dcg = self._dcg(retrieved, relevant)
            idcg = self._idcg(relevant)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg

        return total_ndcg / len(retrieved_ids)

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        return 0.0


class MRRMetric(BaseModel):
    """MRR (Mean Reciprocal Rank) 指标"""

    @property
    def name(self) -> str:
        return "mrr"

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        if not retrieved_ids:
            return 0.0

        total_mrr = 0.0
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            rr = 0.0
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            total_mrr += rr

        return total_mrr / len(retrieved_ids)

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        return 0.0


class MAPMetric(BaseModel):
    """MAP (Mean Average Precision) 指标"""

    @property
    def name(self) -> str:
        return "map"

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        if not retrieved_ids:
            return 0.0

        total_ap = 0.0
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            ap = 0.0
            hits = 0
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    hits += 1
                    ap += hits / (i + 1)
            ap = ap / len(relevant_set) if relevant_set else 0.0
            total_ap += ap

        return total_ap / len(retrieved_ids)

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        return 0.0


__all__ = [
    "BaseMetric",
    "MetricInput",
    "PrecisionMetric",
    "RecallMetric",
    "NDCGMetric",
    "MRRMetric",
    "MAPMetric",
]
