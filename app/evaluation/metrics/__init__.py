"""评估指标模块

提供各种评估指标计算功能。

指标:
- Precision@K: 精确率
- Recall@K: 召回率
- NDCG@K: 归一化折损累积增益
- MRR: 平均倒数排名
- MAP: 平均精确率
- BLEU: BLEU 分数
- ROUGE: ROUGE 分数
"""

from app.evaluation.metrics.base import (
    BaseMetric,
    MAPMetric,
    MetricInput,
    MRRMetric,
    NDCGMetric,
    PrecisionMetric,
    RecallMetric,
)
from app.evaluation.metrics.bleu import BLEUMetric
from app.evaluation.metrics.rouge import RougeMetric, RougeMETRIC

__all__ = [
    # 基类
    "BaseMetric",
    "MetricInput",
    # 检索指标
    "PrecisionMetric",
    "RecallMetric",
    "NDCGMetric",
    "MRRMetric",
    "MAPMetric",
    # 生成指标
    "BLEUMetric",
    "RougeMetric",
    "RougeMETRIC",
]
