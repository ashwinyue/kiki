"""评估模块

提供离线评估功能，支持检索和生成指标计算。

指标:
- 检索: Precision, Recall, NDCG, MRR, MAP
- 生成: BLEU, ROUGE-1, ROUGE-2, ROUGE-L

使用示例:
    ```python
    from app.evaluation import EvaluationService, RunEvaluationRequest

    service = EvaluationService()
    result = await service.run_evaluation(
        RunEvaluationRequest(dataset_name="sample")
    )
    ```
"""

from app.evaluation.dataset import DatasetService, Dataset, get_dataset_service
from app.evaluation.service import EvaluationService, get_evaluation_service
from app.evaluation.types import (
    EvaluationDetail,
    EvaluationEntry,
    EvaluationReport,
    EvaluationStatus,
    EvaluationStatusResponse,
    GenerationMetrics,
    QAPair,
    RetrievalMetrics,
    RunEvaluationRequest,
    RunEvaluationResponse,
)

__all__ = [
    # 数据集
    "DatasetService",
    "Dataset",
    "get_dataset_service",
    # 服务
    "EvaluationService",
    "get_evaluation_service",
    # 类型
    "EvaluationStatus",
    "RunEvaluationRequest",
    "RunEvaluationResponse",
    "EvaluationStatusResponse",
    "EvaluationDetail",
    "EvaluationEntry",
    "EvaluationReport",
    "RetrievalMetrics",
    "GenerationMetrics",
    "QAPair",
]
