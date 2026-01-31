"""评估类型定义

对齐 WeKnora99 评估相关类型。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationStatus(str, Enum):
    """评估状态"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class EvaluationMetric(str, Enum):
    """评估指标"""

    # 检索指标
    PRECISION = "precision"
    RECALL = "recall"
    NDCG = "ndcg"
    MRR = "mrr"
    MAP = "map"

    # 生成指标
    BLEU = "bleu"
    ROUGE_1 = "rouge-1"
    ROUGE_2 = "rouge-2"
    ROUGE_L = "rouge-l"


# ============== 评估请求/响应 ==============


class RunEvaluationRequest(BaseModel):
    """运行评估请求"""

    dataset_name: str = Field(..., description="数据集名称")
    evaluators: list[str] = Field(default=["response_quality"], description="评估器列表")
    agent_type: str = Field(default="chat", description="Agent 类型")
    session_id_prefix: str = Field(default="eval-", description="会话 ID 前缀")
    max_entries: int | None = Field(None, description="最大评估条目数")
    categories: list[str] | None = Field(None, description="评估类别")


class RunEvaluationResponse(BaseModel):
    """运行评估响应"""

    run_id: str = Field(..., description="评估运行 ID")
    status: str = Field(..., description="状态")
    message: str = Field(default="")


class EvaluationStatusResponse(BaseModel):
    """评估状态响应"""

    run_id: str = Field(..., description="评估运行 ID")
    status: str = Field(..., description="状态")
    report: dict[str, Any] | None = Field(None, description="评估报告")


# ============== 评估结果 ==============


class RetrievalMetrics(BaseModel):
    """检索评估指标"""

    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_1: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision@1": self.precision_at_1,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "ndcg@1": self.ndcg_at_1,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "mrr": self.mrr,
            "map": self.map_score,
        }


class GenerationMetrics(BaseModel):
    """生成评估指标"""

    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bleu-1": self.bleu_1,
            "bleu-2": self.bleu_2,
            "bleu-4": self.bleu_4,
            "rouge-1": self.rouge_1,
            "rouge-2": self.rouge_2,
            "rouge-l": self.rouge_l,
        }


class EvaluationEntry(BaseModel):
    """评估条目"""

    id: str
    query: str
    ground_truth: str
    generated_response: str
    retrieved_docs: list[str] = Field(default_factory=list)
    relevant_docs: list[str] = Field(default_factory=list)

    retrieval_metrics: dict[str, float] | None = None
    generation_metrics: dict[str, float] | None = None


class EvaluationDetail(BaseModel):
    """评估详情"""

    run_id: str
    status: EvaluationStatus
    total: int = 0
    finished: int = 0

    retrieval_metrics: dict[str, float] | None = None
    generation_metrics: dict[str, float] | None = None

    entries: list[EvaluationEntry] = Field(default_factory=list)


# ============== 评估报告 ==============


class EvaluatorSummary(BaseModel):
    """评估器汇总"""

    name: str
    score: float
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """评估报告"""

    run_id: str
    created_at: datetime
    total_entries: int

    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)

    evaluator_summaries: list[EvaluatorSummary] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "total_entries": self.total_entries,
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "evaluator_summaries": [
                {"name": s.name, "score": s.score, "details": s.details}
                for s in self.evaluator_summaries
            ],
        }


# ============== 数据集类型 ==============


class QAPair(BaseModel):
    """问答对

    对齐 WeKnora99 QAPair
    """

    qid: int = Field(..., description="问题 ID")
    question: str = Field(..., description="问题")
    pids: list[int] = Field(default_factory=list, description="相关文档 ID 列表")
    passages: list[str] = Field(default_factory=list, description="相关文档内容")
    aid: int = Field(default=0, description="答案 ID")
    answer: str = Field(default="", description="答案")

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "pids": self.pids,
            "passages": self.passages,
            "aid": self.aid,
            "answer": self.answer,
        }


__all__ = [
    "EvaluationStatus",
    "EvaluationMetric",
    "RunEvaluationRequest",
    "RunEvaluationResponse",
    "EvaluationStatusResponse",
    "RetrievalMetrics",
    "GenerationMetrics",
    "EvaluationEntry",
    "EvaluationDetail",
    "EvaluatorSummary",
    "EvaluationReport",
    "QAPair",
]
