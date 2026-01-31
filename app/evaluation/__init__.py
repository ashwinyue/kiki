"""评估框架模块

提供 Agent 评估能力，基于 LangChain Evaluator。

使用示例:
    ```python
    from app.evaluation import (
        ResponseEvaluator,
        ToolCallEvaluator,
        EvaluationRunner,
        create_evaluation_runner,
    )

    # 创建评估运行器
    runner = create_evaluation_runner()

    # 运行评估
    result = await runner.evaluate(
        agent=agent,
        dataset="basic_qa",
        evaluators=["response_quality", "tool_call_accuracy"],
    )

    print(result.summary)
    ```
"""

from app.evaluation.datasets import (
    Dataset,
    DatasetEntry,
    builtin_datasets,
    get_dataset,
    list_datasets,
    register_dataset,
)
from app.evaluation.evaluators import (
    BaseEvaluator,
    ConversationEvaluator,
    ResponseEvaluator,
    ToolCallEvaluator,
    create_evaluator,
)
from app.evaluation.report import (
    EvaluationReport,
    Metric,
    MetricType,
)
from app.evaluation.runner import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationRunner,
    create_evaluation_runner,
)

__all__ = [
    # 评估器
    "BaseEvaluator",
    "ResponseEvaluator",
    "ToolCallEvaluator",
    "ConversationEvaluator",
    "create_evaluator",
    # 数据集
    "Dataset",
    "DatasetEntry",
    "builtin_datasets",
    "get_dataset",
    "list_datasets",
    "register_dataset",
    # 运行器
    "EvaluationRunner",
    "EvaluationConfig",
    "EvaluationResult",
    "create_evaluation_runner",
    # 报告
    "EvaluationReport",
    "Metric",
    "MetricType",
]
