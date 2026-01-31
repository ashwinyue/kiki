"""评估运行器模块

执行评估并生成报告。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from app.agent import get_agent
from app.evaluation.datasets import Dataset, get_dataset
from app.evaluation.evaluators import (
    BaseEvaluator,
    create_evaluator,
)
from app.evaluation.report import (
    EntryResult,
    EvaluationReport,
    create_report,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 配置模型 ===============


class EvaluationConfig(BaseModel):
    """评估配置"""

    dataset_name: str = Field(..., description="数据集名称")
    evaluators: list[str] = Field(
        default=["response_quality"],
        description="评估器列表",
    )
    agent_type: str = Field(default="chat", description="Agent 类型")
    session_id_prefix: str = Field(default="eval-", description="会话 ID 前缀")
    concurrent: bool = Field(default=False, description="是否并发执行")
    max_entries: int | None = Field(None, description="最大条目数限制")
    categories: list[str] | None = Field(None, description="筛选类别")


class EvaluationResult(BaseModel):
    """评估结果

    包含报告和可选的流式更新。
    """

    report: EvaluationReport = Field(..., description="评估报告")
    run_id: str = Field(..., description="运行 ID")

    class Config:
        """Pydantic 配置"""

        arbitrary_types_allowed = True


# ============== 评估运行器 ===============


class EvaluationRunner:
    """评估运行器

    负责执行评估任务并生成报告。
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        evaluators: list[BaseEvaluator] | None = None,
    ):
        """初始化评估运行器

        Args:
            config: 评估配置
            evaluators: 评估器列表
        """
        self._config = config or EvaluationConfig(dataset_name="basic_qa")
        self._custom_evaluators: dict[str, BaseEvaluator] = {e.name: e for e in (evaluators or [])}
        self._agent: Any | None = None

    async def _get_agent(self) -> Any:
        """获取 Agent 实例

        Returns:
            Agent 实例
        """
        if self._agent is None:
            self._agent = await get_agent()
        return self._agent

    def _get_evaluator(self, name: str) -> BaseEvaluator:
        """获取评估器实例

        Args:
            name: 评估器名称

        Returns:
            评估器实例
        """
        if name not in self._custom_evaluators:
            self._custom_evaluators[name] = create_evaluator(name)
        return self._custom_evaluators[name]

    async def run(
        self,
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """运行评估

        Args:
            config: 评估配置（覆盖初始化配置）

        Returns:
            EvaluationResult: 评估结果
        """
        effective_config = config or self._config
        run_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            "evaluation_run_start",
            run_id=run_id,
            dataset=effective_config.dataset_name,
            evaluators=effective_config.evaluators,
        )

        try:
            # 获取数据集
            dataset = get_dataset(effective_config.dataset_name)
            if dataset is None:
                raise ValueError(f"数据集不存在: {effective_config.dataset_name}")

            # 筛选条目
            entries = self._filter_entries(dataset, effective_config)

            # 限制条目数
            if effective_config.max_entries:
                entries = entries[: effective_config.max_entries]

            # 获取 Agent
            agent = await self._get_agent()

            # 执行评估
            entry_results = []
            for i, entry in enumerate(entries):
                logger.debug(
                    "evaluating_entry",
                    run_id=run_id,
                    entry_index=i,
                    entry_description=entry.description,
                )

                result = await self._evaluate_entry(
                    agent=agent,
                    entry=entry,
                    config=effective_config,
                    run_id=run_id,
                )
                entry_results.append(result)

            # 生成报告
            duration = time.time() - start_time
            report = create_report(
                run_id=run_id,
                agent_name=effective_config.agent_type,
                dataset_name=effective_config.dataset_name,
                evaluators=effective_config.evaluators,
                entry_results=entry_results,
                config=effective_config.model_dump(),
                duration_seconds=duration,
            )

            logger.info(
                "evaluation_run_complete",
                run_id=run_id,
                pass_rate=report.overall_pass_rate,
                score=report.overall_score,
                duration=duration,
            )

            return EvaluationResult(report=report, run_id=run_id)

        except Exception:
            logger.exception("evaluation_run_failed", run_id=run_id)
            # 返回错误报告
            duration = time.time() - start_time
            return EvaluationResult(
                report=create_report(
                    run_id=run_id,
                    agent_name=effective_config.agent_type,
                    dataset_name=effective_config.dataset_name,
                    evaluators=effective_config.evaluators,
                    entry_results=[],
                    config=effective_config.model_dump(),
                    duration_seconds=duration,
                ),
                run_id=run_id,
            )

    async def run_stream(
        self,
        config: EvaluationConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式运行评估

        Args:
            config: 评估配置

        Yields:
            评估进度更新
        """
        effective_config = config or self._config
        run_id = str(uuid.uuid4())
        start_time = time.time()

        yield {
            "type": "start",
            "run_id": run_id,
            "dataset": effective_config.dataset_name,
            "evaluators": effective_config.evaluators,
        }

        try:
            dataset = get_dataset(effective_config.dataset_name)
            if dataset is None:
                raise ValueError(f"数据集不存在: {effective_config.dataset_name}")

            entries = self._filter_entries(dataset, effective_config)
            if effective_config.max_entries:
                entries = entries[: effective_config.max_entries]

            agent = await self._get_agent()

            entry_results = []
            for i, entry in enumerate(entries):
                yield {
                    "type": "progress",
                    "run_id": run_id,
                    "current": i + 1,
                    "total": len(entries),
                    "description": entry.description,
                }

                result = await self._evaluate_entry(
                    agent=agent,
                    entry=entry,
                    config=effective_config,
                    run_id=run_id,
                )
                entry_results.append(result)

                yield {
                    "type": "entry_result",
                    "run_id": run_id,
                    "entry_index": i,
                    "passed": result.passed,
                    "score": result.score,
                }

            duration = time.time() - start_time
            report = create_report(
                run_id=run_id,
                agent_name=effective_config.agent_type,
                dataset_name=effective_config.dataset_name,
                evaluators=effective_config.evaluators,
                entry_results=entry_results,
                config=effective_config.model_dump(),
                duration_seconds=duration,
            )

            yield {
                "type": "complete",
                "run_id": run_id,
                "report": report.to_dict(),
            }

        except Exception as e:
            logger.exception("evaluation_stream_failed", run_id=run_id)
            yield {
                "type": "error",
                "run_id": run_id,
                "error": str(e),
            }

    async def _evaluate_entry(
        self,
        agent: Any,
        entry: Any,
        config: EvaluationConfig,
        run_id: str,
    ) -> EntryResult:
        """评估单个条目

        Args:
            agent: Agent 实例
            entry: 数据集条目
            config: 评估配置
            run_id: 运行 ID

        Returns:
            EntryResult: 条目结果
        """
        session_id = f"{config.session_id_prefix}{run_id[:8]}"

        try:
            # 调用 Agent
            messages = await agent.get_response(
                message=entry.input_data.get("message", ""),
                session_id=session_id,
                user_id=f"eval-{run_id}",
            )

            # 提取响应
            response = ""
            tool_calls = []
            for msg in messages:
                if hasattr(msg, "content"):
                    if msg.content:
                        response += str(msg.content)
                if hasattr(msg, "tool_calls"):
                    for tc in msg.tool_calls:
                        tool_calls.append(
                            {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            }
                        )

            output_data = {
                "response": response,
                "messages": [str(m) for m in messages],
                "tool_calls": tool_calls,
            }

            # 运行评估器
            evaluator_results = []
            all_passed = True
            total_score = 0.0

            for evaluator_name in config.evaluators:
                evaluator = self._get_evaluator(evaluator_name)
                result = await evaluator.evaluate(
                    input_data=entry.input_data,
                    output_data=output_data,
                    expected=entry.expected,
                    context=entry.metadata,
                )
                evaluator_results.append(result.model_dump())
                all_passed = all_passed and result.passed
                total_score += result.score

            avg_score = total_score / len(config.evaluators) if config.evaluators else 0.0

            return EntryResult(
                entry_id=str(id(entry)),
                category=entry.category,
                description=entry.description,
                passed=all_passed,
                score=avg_score,
                evaluator_results=evaluator_results,
                input_data=entry.input_data,
                output_data=output_data,
            )

        except Exception as e:
            logger.exception(
                "entry_evaluation_failed",
                run_id=run_id,
                entry_description=entry.description,
            )
            return EntryResult(
                entry_id=str(id(entry)),
                category=entry.category,
                description=entry.description,
                passed=False,
                score=0.0,
                error=str(e),
            )

    def _filter_entries(
        self,
        dataset: Dataset,
        config: EvaluationConfig,
    ) -> list[Any]:
        """筛选数据集条目

        Args:
            dataset: 数据集
            config: 评估配置

        Returns:
            筛选后的条目列表
        """
        entries = dataset.entries

        if config.categories:
            entries = [e for e in entries if e.category and e.category in config.categories]

        return entries


# ============== 便捷函数 ===============


def create_evaluation_runner(
    config: EvaluationConfig | None = None,
    evaluators: list[BaseEvaluator] | None = None,
) -> EvaluationRunner:
    """创建评估运行器

    Args:
        config: 评估配置
        evaluators: 自定义评估器列表

    Returns:
        EvaluationRunner 实例

    Examples:
        ```python
        from app.evaluation import create_evaluation_runner, EvaluationConfig

        runner = create_evaluation_runner(
            config=EvaluationConfig(
                dataset_name="basic_qa",
                evaluators=["response_quality", "tool_call_accuracy"],
            )
        )

        result = await runner.run()
        print(result.report.to_markdown())
        ```
    """
    return EvaluationRunner(config=config, evaluators=evaluators)
