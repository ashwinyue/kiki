"""评估报告模块

生成和格式化评估报告。
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============== 报告模型 ===============


class MetricType(str, Enum):
    """指标类型"""

    SCORE = "score"  # 评分
    PASS_RATE = "pass_rate"  # 通过率
    COUNT = "count"  # 计数
    PERCENTILE = "percentile"  # 百分位


class Metric(BaseModel):
    """评估指标"""

    name: str = Field(..., description="指标名称")
    value: float = Field(..., description="指标值")
    type: MetricType = Field(..., description="指标类型")
    description: str = Field(..., description="指标描述")
    unit: str | None = Field(None, description="单位")


class EntryResult(BaseModel):
    """单条测试结果"""

    entry_id: str | None = Field(None, description="条目ID")
    category: str | None = Field(None, description="类别")
    description: str | None = Field(None, description="描述")
    passed: bool = Field(..., description="是否通过")
    score: float = Field(..., description="综合评分")
    evaluator_results: list[dict[str, Any]] = Field(default_factory=list, description="评估器结果")
    input_data: dict[str, Any] = Field(default_factory=dict, description="输入数据")
    output_data: dict[str, Any] = Field(default_factory=dict, description="输出数据")
    error: str | None = Field(None, description="错误信息")


class EvaluatorSummary(BaseModel):
    """评估器汇总"""

    name: str = Field(..., description="评估器名称")
    description: str = Field(..., description="评估器描述")
    total_count: int = Field(..., description="总评估次数")
    pass_count: int = Field(..., description="通过次数")
    fail_count: int = Field(..., description="失败次数")
    pass_rate: float = Field(..., description="通过率")
    avg_score: float = Field(..., description="平均评分")
    min_score: float = Field(..., description="最低评分")
    max_score: float = Field(..., description="最高评分")


class EvaluationReport(BaseModel):
    """评估报告

    包含评估结果的完整报告。
    """

    # 基本信息
    run_id: str = Field(..., description="运行ID")
    timestamp: str = Field(..., description="时间戳")
    agent_name: str = Field(..., description="Agent名称")
    dataset_name: str = Field(..., description="数据集名称")

    # 评估配置
    evaluators: list[str] = Field(..., description="使用的评估器")
    config: dict[str, Any] = Field(default_factory=dict, description="配置参数")

    # 结果汇总
    total_entries: int = Field(..., description="总条目数")
    passed_entries: int = Field(..., description="通过条目数")
    failed_entries: int = Field(..., description="失败条目数")
    overall_pass_rate: float = Field(..., description="总体通过率")
    overall_score: float = Field(..., description="总体评分")

    # 评估器汇总
    evaluator_summaries: list[EvaluatorSummary] = Field(
        default_factory=list, description="评估器汇总"
    )

    # 详细结果
    entry_results: list[EntryResult] = Field(default_factory=list, description="条目结果")

    # 指标
    metrics: list[Metric] = Field(default_factory=list, description="额外指标")

    # 执行信息
    duration_seconds: float = Field(0.0, description="执行时长（秒）")
    error: str | None = Field(None, description="错误信息")

    class Config:
        """Pydantic 配置"""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Returns:
            字典表示
        """
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串

        Args:
            indent: 缩进空格数

        Returns:
            JSON 字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式

        Returns:
            Markdown 字符串
        """
        lines = [
            "# 评估报告",
            "",
            "## 基本信息",
            f"- **运行ID**: `{self.run_id}`",
            f"- **时间**: {self.timestamp}",
            f"- **Agent**: {self.agent_name}",
            f"- **数据集**: {self.dataset_name}",
            f"- **执行时长**: {self.duration_seconds:.2f}s",
            "",
            "## 评估配置",
            f"- **评估器**: {', '.join(self.evaluators)}",
            "",
            "## 结果汇总",
            "",
            "| 指标 | 值 |",
            "|------|------|",
            f"| 总条目数 | {self.total_entries} |",
            f"| 通过条目 | {self.passed_entries} |",
            f"| 失败条目 | {self.failed_entries} |",
            f"| **通过率** | **{self.overall_pass_rate:.1%}** |",
            f"| **总体评分** | **{self.overall_score:.3f}** |",
            "",
        ]

        # 评估器汇总
        if self.evaluator_summaries:
            lines.extend(
                [
                    "## 评估器详情",
                    "",
                    "| 评估器 | 通过率 | 平均评分 | 范围 |",
                    "|--------|--------|----------|------|",
                ]
            )
            for summary in self.evaluator_summaries:
                lines.append(
                    f"| {summary.name} | {summary.pass_rate:.1%} | "
                    f"{summary.avg_score:.3f} | "
                    f"{summary.min_score:.2f} - {summary.max_score:.2f} |"
                )
            lines.append("")

        # 指标
        if self.metrics:
            lines.extend(
                [
                    "## 详细指标",
                    "",
                ]
            )
            for metric in self.metrics:
                unit_str = f" {metric.unit}" if metric.unit else ""
                lines.append(
                    f"- **{metric.name}**: {metric.value}{unit_str} - {metric.description}"
                )
            lines.append("")

        # 详细结果
        if self.entry_results:
            lines.extend(
                [
                    "## 详细结果",
                    "",
                ]
            )
            for i, result in enumerate(self.entry_results, 1):
                status_icon = "✅" if result.passed else "❌"
                lines.append(f"### {status_icon} 测试 #{i}: {result.description or 'N/A'}")
                if result.category:
                    lines.append(f"**类别**: {result.category}")
                lines.append(f"**评分**: {result.score:.3f}")
                if result.error:
                    lines.append(f"**错误**: `{result.error}`")
                lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """获取摘要信息

        Returns:
            摘要字符串
        """
        return (
            f"评估报告 {self.run_id[:8]}: "
            f"{self.passed_entries}/{self.total_entries} 通过 "
            f"({self.overall_pass_rate:.1%}), "
            f"评分 {self.overall_score:.3f}"
        )


def create_report(
    run_id: str,
    agent_name: str,
    dataset_name: str,
    evaluators: list[str],
    entry_results: list[EntryResult],
    config: dict[str, Any] | None = None,
    duration_seconds: float = 0.0,
) -> EvaluationReport:
    """创建评估报告

    Args:
        run_id: 运行ID
        agent_name: Agent名称
        dataset_name: 数据集名称
        evaluators: 评估器列表
        entry_results: 条目结果列表
        config: 配置参数
        duration_seconds: 执行时长

    Returns:
        EvaluationReport 实例
    """
    # 计算总体统计
    total_entries = len(entry_results)
    passed_entries = sum(1 for r in entry_results if r.passed)
    failed_entries = total_entries - passed_entries
    overall_pass_rate = passed_entries / total_entries if total_entries > 0 else 0.0
    overall_score = (
        sum(r.score for r in entry_results) / total_entries if total_entries > 0 else 0.0
    )

    # 按评估器分组统计
    evaluator_stats: dict[str, list[dict[str, Any]]] = {}
    for result in entry_results:
        for eval_result in result.evaluator_results:
            name = eval_result.get("evaluator_name", "unknown")
            if name not in evaluator_stats:
                evaluator_stats[name] = []
            evaluator_stats[name].append(eval_result)

    evaluator_summaries = []
    for name, results in evaluator_stats.items():
        scores = [r.get("score", 0.0) for r in results]
        pass_count = sum(1 for r in results if r.get("passed", False))
        evaluator_summaries.append(
            EvaluatorSummary(
                name=name,
                description=name,  # 可以从评估器获取
                total_count=len(results),
                pass_count=pass_count,
                fail_count=len(results) - pass_count,
                pass_rate=pass_count / len(results) if results else 0.0,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
            )
        )

    # 生成额外指标
    metrics = [
        Metric(
            name="overall_pass_rate",
            value=overall_pass_rate,
            type=MetricType.PASS_RATE,
            description="总体通过率",
            unit="%",
        ),
        Metric(
            name="overall_score",
            value=overall_score,
            type=MetricType.SCORE,
            description="总体评分",
        ),
    ]

    return EvaluationReport(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        agent_name=agent_name,
        dataset_name=dataset_name,
        evaluators=evaluators,
        config=config or {},
        total_entries=total_entries,
        passed_entries=passed_entries,
        failed_entries=failed_entries,
        overall_pass_rate=overall_pass_rate,
        overall_score=overall_score,
        evaluator_summaries=evaluator_summaries,
        entry_results=entry_results,
        metrics=metrics,
        duration_seconds=duration_seconds,
    )
