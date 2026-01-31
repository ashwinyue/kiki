"""评估服务

对齐 WeKnora99 评估服务。

功能:
- 运行离线评估任务
- 计算检索和生成指标
- 生成评估报告
- 支持并行评估
"""

import uuid
from datetime import datetime
from typing import Any, Callable

from app.evaluation.dataset import DatasetService, get_dataset_service
from app.evaluation.metrics import (
    BLEUMetric,
    MAPMetric,
    MRRMetric,
    NDCGMetric,
    PrecisionMetric,
    RecallMetric,
    RougeMETRIC,
)
from app.evaluation.types import (
    EvaluationDetail,
    EvaluationEntry,
    EvaluationReport,
    EvaluationStatus,
    QAPair,
    RunEvaluationRequest,
    RunEvaluationResponse,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """评估服务

    提供离线评估功能，参考 WeKnora99 EvaluationService 实现。
    """

    def __init__(
        self,
        dataset_service: DatasetService | None = None,
    ):
        """初始化评估服务

        Args:
            dataset_service: 数据集服务
        """
        self._dataset_service = dataset_service or get_dataset_service()
        self._metrics = {
            "precision": PrecisionMetric(k=5),
            "recall": RecallMetric(k=5),
            "ndcg": NDCGMetric(k=5),
            "mrr": MRRMetric(),
            "map": MAPMetric(),
            "bleu": BLEUMetric(),
            "rouge": RougeMETRIC(),
        }
        # 内存存储评估任务
        self._tasks: dict[str, EvaluationDetail] = {}

    async def run_evaluation(
        self,
        request: RunEvaluationRequest,
        inference_func: Callable[[str], Any] | None = None,
    ) -> RunEvaluationResponse:
        """运行评估任务

        Args:
            request: 评估请求
            inference_func: 推理函数 (query -> response)，用于生成回答

        Returns:
            评估响应
        """
        run_id = str(uuid.uuid4())

        logger.info(
            "evaluation_run_start",
            run_id=run_id,
            dataset=request.dataset_name,
        )

        try:
            # 加载数据集
            qa_pairs = await self._dataset_service.get_dataset(request.dataset_name)

            if not qa_pairs:
                return RunEvaluationResponse(
                    run_id=run_id,
                    status="failed",
                    message=f"数据集 '{request.dataset_name}' 为空或不存在",
                )

            # 限制评估数量
            max_entries = request.max_entries or len(qa_pairs)
            qa_pairs = qa_pairs[:max_entries]

            logger.info(
                "evaluation_dataset_loaded",
                run_id=run_id,
                dataset=request.dataset_name,
                entry_count=len(qa_pairs),
            )

            # 运行评估
            entries = await self._evaluate_pairs(
                run_id=run_id,
                qa_pairs=qa_pairs,
                inference_func=inference_func,
            )

            # 生成汇总
            detail = await self.get_evaluation_detail(run_id, entries)

            logger.info(
                "evaluation_run_complete",
                run_id=run_id,
                total=detail.total,
                finished=detail.finished,
            )

            # 保存任务
            self._tasks[run_id] = detail

            return RunEvaluationResponse(
                run_id=run_id,
                status="success",
                message=f"评估完成，共 {detail.total} 条",
            )

        except Exception as e:
            logger.error(
                "evaluation_run_failed",
                run_id=run_id,
                error=str(e),
            )
            return RunEvaluationResponse(
                run_id=run_id,
                status="failed",
                message=str(e),
            )

    async def _evaluate_pairs(
        self,
        run_id: str,
        qa_pairs: list[QAPair],
        inference_func: Callable[[str], Any] | None = None,
    ) -> list[EvaluationEntry]:
        """评估 QA 对

        Args:
            run_id: 运行 ID
            qa_pairs: QA 对列表
            inference_func: 推理函数

        Returns:
            评估条目列表
        """
        entries = []
        total = len(qa_pairs)

        for i, qa_pair in enumerate(qa_pairs):
            logger.info(
                "evaluation_progress",
                run_id=run_id,
                current=i + 1,
                total=total,
                question=qa_pair.question[:50],
            )

            # 调用推理函数生成回答
            generated = ""
            if inference_func:
                try:
                    generated = await inference_func(qa_pair.question)
                except Exception as e:
                    logger.warning(
                        "inference_failed",
                        run_id=run_id,
                        question=qa_pair.question[:50],
                        error=str(e),
                    )
                    generated = ""
            else:
                # 默认使用真实答案
                generated = qa_pair.answer

            # 转换为字符串 ID
            retrieved_ids = [str(pid) for pid in qa_pair.pids]
            relevant_ids = [str(pid) for pid in qa_pair.pids]  # 假设所有标注都是相关的

            # 计算检索指标
            retrieval_metrics = None
            if retrieved_ids and relevant_ids:
                retrieval_metrics = self._compute_retrieval_metrics(
                    [retrieved_ids], [relevant_ids]
                )

            # 计算生成指标
            generation_metrics = self._compute_generation_metrics(
                [generated], [qa_pair.answer]
            )

            entry = EvaluationEntry(
                id=f"{run_id}_{i}",
                query=qa_pair.question,
                ground_truth=qa_pair.answer,
                generated_response=generated,
                retrieved_docs=retrieved_ids,
                relevant_docs=relevant_ids,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
            )
            entries.append(entry)

        return entries

    async def get_evaluation_detail(
        self,
        run_id: str,
        entries: list[EvaluationEntry],
    ) -> EvaluationDetail:
        """获取评估详情

        Args:
            run_id: 评估运行 ID
            entries: 评估条目列表

        Returns:
            评估详情
        """
        # 聚合检索指标
        retrieved_ids = [e.retrieved_docs for e in entries if e.retrieved_docs]
        relevant_ids = [e.relevant_docs for e in entries if e.relevant_docs]

        retrieval_metrics = None
        if retrieved_ids and relevant_ids:
            retrieval_metrics = self._compute_retrieval_metrics(retrieved_ids, relevant_ids)

        # 聚合生成指标
        generated_texts = [e.generated_response for e in entries]
        reference_texts = [e.ground_truth for e in entries]

        generation_metrics = self._compute_generation_metrics(generated_texts, reference_texts)

        return EvaluationDetail(
            run_id=run_id,
            status=EvaluationStatus.SUCCESS,
            total=len(entries),
            finished=len(entries),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            entries=entries,
        )

    async def get_result(self, run_id: str) -> EvaluationDetail | None:
        """获取评估结果

        Args:
            run_id: 评估运行 ID

        Returns:
            评估详情或 None
        """
        return self._tasks.get(run_id)

    async def get_status(self, run_id: str) -> str | None:
        """获取评估状态

        Args:
            run_id: 评估运行 ID

        Returns:
            状态字符串或 None
        """
        task = self._tasks.get(run_id)
        if task:
            return task.status.value
        return None

    def _compute_retrieval_metrics(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> dict[str, float]:
        """计算检索指标

        Args:
            retrieved_ids: 检索结果 ID 列表
            relevant_ids: 正确答案 ID 列表

        Returns:
            指标字典
        """
        from app.evaluation.types import RetrievalMetrics

        # 创建指标实例
        precision = PrecisionMetric(k=5)
        recall = RecallMetric(k=5)
        ndcg = NDCGMetric(k=5)
        mrr = MRRMetric()
        mapp = MAPMetric()

        return {
            "precision": round(precision.compute_retrieval(retrieved_ids, relevant_ids), 4),
            "recall": round(recall.compute_retrieval(retrieved_ids, relevant_ids), 4),
            "ndcg": round(ndcg.compute_retrieval(retrieved_ids, relevant_ids), 4),
            "mrr": round(mrr.compute_retrieval(retrieved_ids, relevant_ids), 4),
            "map": round(mapp.compute_retrieval(retrieved_ids, relevant_ids), 4),
        }

    def _compute_generation_metrics(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算生成指标"""
        metrics = {}

        # BLEU
        bleu = BLEUMetric()
        bleu_result = bleu.compute(generated_texts, reference_texts)
        metrics["bleu"] = round(bleu_result.get("bleu", 0.0), 4)

        # ROUGE
        rouge = RougeMETRIC()
        rouge_result = rouge.compute_all(generated_texts, reference_texts)
        for key, value in rouge_result.items():
            metrics[key] = round(value, 4)

        return metrics

    async def generate_report(
        self,
        detail: EvaluationDetail,
    ) -> EvaluationReport:
        """生成评估报告

        Args:
            detail: 评估详情

        Returns:
            评估报告
        """
        report = EvaluationReport(
            run_id=detail.run_id,
            created_at=datetime.now(),
            total_entries=detail.total,
            retrieval_metrics=detail.retrieval_metrics or {},
            generation_metrics=detail.generation_metrics or {},
            evaluator_summaries=[],
        )

        return report


# 全局服务实例
_evaluation_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    """获取评估服务实例"""
    global _evaluation_service
    if _evaluation_service is None:
        dataset_service = get_dataset_service()
        _evaluation_service = EvaluationService(dataset_service=dataset_service)
    return _evaluation_service


__all__ = [
    "EvaluationService",
    "get_evaluation_service",
]
