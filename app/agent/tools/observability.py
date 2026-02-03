"""
工具可观测性模块

为工具执行提供监控、追踪和健康度报告功能。

集成 Prometheus 指标：
- tool_calls_total: 工具调用计数
- tool_duration_seconds: 工具执行时长
- tool_errors_total: 工具错误计数
- tool_slow_calls: 慢调用告警
"""

import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool
from prometheus_client import Counter, Gauge

from app.observability.logging import get_logger
from app.observability.metrics import (
    tool_calls_total,
    tool_duration_seconds,
)

logger = get_logger(__name__)

# 额外指标

# 工具错误计数（单独的 Counter，便于告警）
tool_errors_total = Counter(
    "tool_errors_total",
    "工具错误总数",
    ["tool_name", "error_type"],
)

# 慢调用计数（超过阈值）
tool_slow_calls_total = Counter(
    "tool_slow_calls_total",
    "工具慢调用总数（超过阈值）",
    ["tool_name", "threshold_seconds"],
)

# 当前正在执行的工具（活跃追踪）
tool_active_gauge = Gauge(
    "tool_active",
    "正在执行的工具数",
    ["tool_name"],
)

# 工具健康度指标（成功率）
tool_health_score = Gauge(
    "tool_health_score",
    "工具健康度评分（0-100）",
    ["tool_name"],
)


# 工具执行统计

class ToolStats:
    """工具统计收集器

    跟踪每个工具的执行统计，用于健康度评分。
    """

    def __init__(self) -> None:
        # 调用统计
        self.call_counts: defaultdict[str, int] = defaultdict(int)
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.slow_counts: defaultdict[str, int] = defaultdict(int)

        # 耗时统计
        self.total_duration: defaultdict[str, float] = defaultdict(float)
        self.max_duration: defaultdict[str, float] = defaultdict(float)

        # 最后更新时间
        self.last_updated: dict[str, float] = {}

    def record_call(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        slow_threshold: float = 5.0,
    ) -> None:
        """记录工具调用

        Args:
            tool_name: 工具名称
            duration: 执行时长（秒）
            success: 是否成功
            slow_threshold: 慢调用阈值（秒）
        """
        self.call_counts[tool_name] += 1
        self.total_duration[tool_name] += duration
        self.last_updated[tool_name] = time.time()

        # 更新最大时长
        if duration > self.max_duration[tool_name]:
            self.max_duration[tool_name] = duration

        if not success:
            self.error_counts[tool_name] += 1
        elif duration > slow_threshold:
            self.slow_counts[tool_name] += 1

        # 更新 Prometheus 健康度评分
        self._update_health_score(tool_name)

    def get_stats(self, tool_name: str) -> dict[str, Any]:
        """获取工具统计

        Args:
            tool_name: 工具名称

        Returns:
            统计信息字典
        """
        total_calls = self.call_counts.get(tool_name, 0)
        total_duration = self.total_duration.get(tool_name, 0)
        error_count = self.error_counts.get(tool_name, 0)
        slow_count = self.slow_counts.get(tool_name, 0)

        return {
            "tool_name": tool_name,
            "total_calls": total_calls,
            "error_count": error_count,
            "slow_count": slow_count,
            "success_rate": (total_calls - error_count) / total_calls if total_calls > 0 else 1.0,
            "error_rate": error_count / total_calls if total_calls > 0 else 0.0,
            "slow_rate": slow_count / total_calls if total_calls > 0 else 0.0,
            "avg_duration": total_duration / total_calls if total_calls > 0 else 0,
            "max_duration": self.max_duration.get(tool_name, 0),
            "last_updated": self.last_updated.get(tool_name),
        }

    def get_all_stats(self) -> list[dict[str, Any]]:
        """获取所有工具的统计

        Returns:
            统计信息列表
        """
        all_tools = set(self.call_counts.keys())
        return [self.get_stats(tool) for tool in sorted(all_tools)]

    def _update_health_score(self, tool_name: str) -> None:
        """更新工具健康度评分

        评分标准：
        - 成功率权重 70%
        - 慢调用率权重 30%
        - 最终评分 0-100
        """
        stats = self.get_stats(tool_name)
        total_calls = stats["total_calls"]

        if total_calls < 5:
            # 样本量太小，给默认评分
            health_score = 80.0
        else:
            success_rate = stats["success_rate"]
            slow_rate = stats["slow_rate"]

            # 成功率得分（70% 权重）
            success_score = success_rate * 70

            # 慢调用得分（30% 权重，慢调用率越低越好）
            slow_score = (1 - min(slow_rate, 1.0)) * 30

            health_score = success_score + slow_score

        tool_health_score.labels(tool_name=tool_name).set(max(0, min(100, health_score)))


# 全局统计实例
_tool_stats = ToolStats()


# 监控装饰器和上下文管理器

@asynccontextmanager
async def monitor_tool_execution(
    tool_name: str,
    slow_threshold: float = 5.0,
    track_stats: bool = True,
):
    """工具执行监控上下文管理器

    自动追踪：
    - 调用计数（成功/失败）
    - 执行时长
    - 慢调用检测
    - 健康度评分更新

    Args:
        tool_name: 工具名称
        slow_threshold: 慢调用阈值（秒），默认 5 秒
        track_stats: 是否更新统计（默认 True）

    Yields:
        None

    Example:
        ```python
        async with monitor_tool_execution("search_web"):
            result = await search_function(query)
        ```
    """
    start_time = time.time()
    status = "success"
    error_type = None

    # 标记工具活跃
    tool_active_gauge.labels(tool_name=tool_name).inc()

    try:
        yield
    except Exception as e:
        status = "error"
        error_type = type(e).__name__
        tool_errors_total.labels(tool_name=tool_name, error_type=error_type).inc()
        logger.warning(
            "tool_execution_error",
            tool_name=tool_name,
            error_type=error_type,
            error=str(e),
        )
        raise
    finally:
        duration = time.time() - start_time

        # 记录 Prometheus 指标
        tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        tool_duration_seconds.labels(tool_name=tool_name).observe(duration)

        # 检测慢调用
        if duration > slow_threshold:
            tool_slow_calls_total.labels(
                tool_name=tool_name,
                threshold_seconds=str(slow_threshold),
            ).inc()
            logger.warning(
                "tool_slow_call",
                tool_name=tool_name,
                duration=f"{duration:.2f}s",
                threshold=f"{slow_threshold}s",
            )

        # 更新统计
        if track_stats:
            _tool_stats.record_call(
                tool_name=tool_name,
                duration=duration,
                success=(status == "success"),
                slow_threshold=slow_threshold,
            )

        # 清除活跃标记
        tool_active_gauge.labels(tool_name=tool_name).dec()

        logger.debug(
            "tool_executed",
            tool_name=tool_name,
            status=status,
            duration=f"{duration:.3f}s",
        )


# 工具包装器

class MonitoredTool(BaseTool):
    """带监控的工具包装器

    自动为任何 BaseTool 添加监控功能。

    使用 __getattr__ 委托避免 Pydantic 属性设置问题。
    """

    def __init__(self, original_tool: BaseTool, slow_threshold: float = 5.0):
        """初始化监控工具

        Args:
            original_tool: 原始工具
            slow_threshold: 慢调用阈值（秒）
        """
        # 使用内部属性避免 Pydantic 冲突
        object.__setattr__(self, "_original_tool", original_tool)
        object.__setattr__(self, "_slow_threshold", slow_threshold)

    def __getattr__(self, name: str) -> Any:
        """委托属性访问到原始工具"""
        original_tool = object.__getattribute__(self, "_original_tool")
        return getattr(original_tool, name)

    @property
    def name(self) -> str:
        """工具名称"""
        return self._original_tool.name

    @classmethod
    def wrap(cls, tool: BaseTool, slow_threshold: float = 5.0) -> "MonitoredTool":
        """包装工具

        Args:
            tool: 原始工具
            slow_threshold: 慢调用阈值（秒）

        Returns:
            监控工具实例
        """
        return cls(tool, slow_threshold=slow_threshold)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """异步执行（带监控）

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            工具执行结果
        """
        tool_name = self._original_tool.name
        async with monitor_tool_execution(tool_name, self._slow_threshold):
            return await self._original_tool._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """同步执行（带监控）

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            工具执行结果
        """
        async def _run_async():
            tool_name = self._original_tool.name
            async with monitor_tool_execution(tool_name, self._slow_threshold):
                # 如果原工具有异步方法，使用异步方法
                if hasattr(self._original_tool, "_arun"):
                    return await self._original_tool._arun(*args, **kwargs)
                # 否则在线程池中运行同步方法
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._original_tool._run(*args, **kwargs),
                )

        return asyncio.run(_run_async())


def wrap_tools_with_monitoring(
    tools: list[BaseTool],
    slow_threshold: float = 5.0,
) -> list[BaseTool]:
    """批量包装工具

    Args:
        tools: 原始工具列表
        slow_threshold: 慢调用阈值（秒）

    Returns:
        监控工具列表

    Example:
        ```python
        from app.agent.tools.observability import wrap_tools_with_monitoring

        monitored_tools = wrap_tools_with_monitoring(all_tools)
        agent = create_react_agent(llm, monitored_tools)
        ```
    """
    return [
        MonitoredTool.wrap(tool, slow_threshold=slow_threshold)
        if not isinstance(tool, MonitoredTool)
        else tool
        for tool in tools
    ]


# 健康度报告

def get_tool_health_report(tool_name: str | None = None) -> dict[str, Any]:
    """获取工具健康度报告

    Args:
        tool_name: 工具名称，None 表示获取所有工具

    Returns:
        健康度报告

    Example:
        ```python
        # 获取单个工具报告
        report = get_tool_health_report("search_web")

        # 获取所有工具报告
        all_reports = get_tool_health_report()
        ```
    """
    if tool_name:
        return _tool_stats.get_stats(tool_name)
    return {
        "tools": _tool_stats.get_all_stats(),
        "summary": {
            "total_tools": len(_tool_stats.call_counts),
            "total_calls": sum(_tool_stats.call_counts.values()),
            "total_errors": sum(_tool_stats.error_counts.values()),
        },
    }


def format_health_report(report: dict[str, Any]) -> str:
    """格式化健康度报告为可读文本

    Args:
        report: 健康度报告字典

    Returns:
        格式化的文本
    """
    if "tools" in report:
        # 多工具报告
        lines = ["工具健康度报告", "=" * 50]

        for tool_report in report["tools"]:
            name = tool_report["tool_name"]
            lines.append(f"\n【{name}】")
            lines.append(f"  总调用: {tool_report['total_calls']}")
            lines.append(f"  成功率: {tool_report['success_rate']:.1%}")
            lines.append(f"  慢调用率: {tool_report['slow_rate']:.1%}")
            lines.append(f"  平均耗时: {tool_report['avg_duration']:.3f}s")
            lines.append(f"  最大耗时: {tool_report['max_duration']:.3f}s")

        summary = report.get("summary", {})
        lines.append("\n" + "=" * 50)
        lines.append(f"总工具数: {summary.get('total_tools', 0)}")
        lines.append(f"总调用数: {summary.get('total_calls', 0)}")
        lines.append(f"总错误数: {summary.get('total_errors', 0)}")

        return "\n".join(lines)

    else:
        # 单工具报告
        lines = [f"工具健康度: {report['tool_name']}", "-" * 40]
        lines.append(f"总调用: {report['total_calls']}")
        lines.append(f"成功: {report['success_rate']:.1%}")
        lines.append(f"慢调用: {report['slow_rate']:.1%}")
        lines.append(f"平均耗时: {report['avg_duration']:.3f}s")

        return "\n".join(lines)


# 导出

__all__ = [
    # 上下文管理器
    "monitor_tool_execution",
    # 工具包装器
    "MonitoredTool",
    "wrap_tools_with_monitoring",
    # 健康度报告
    "get_tool_health_report",
    "format_health_report",
    # 统计实例
    "_tool_stats",
    # Prometheus 指标
    "tool_errors_total",
    "tool_slow_calls_total",
    "tool_active_gauge",
    "tool_health_score",
]
