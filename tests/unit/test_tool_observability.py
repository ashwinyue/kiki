"""
工具可观测性模块单元测试

测试工具监控包装器、健康度报告和 Prometheus 指标。
"""

import asyncio
import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import Counter, Gauge, Histogram

from langchain_core.tools import tool

from app.agent.tools.observability import (
    MonitoredTool,
    ToolStats,
    format_health_report,
    get_tool_health_report,
    monitor_tool_execution,
    wrap_tools_with_monitoring,
    _tool_stats,
    tool_active_gauge,
    tool_errors_total,
    tool_health_score,
    tool_slow_calls_total,
)


# ========== 测试隔离 Fixture ==========

@pytest.fixture(autouse=True)
def reset_tool_stats():
    """每个测试前重置全局统计状态"""
    # 清空统计
    _tool_stats.call_counts.clear()
    _tool_stats.error_counts.clear()
    _tool_stats.slow_counts.clear()
    _tool_stats.total_duration.clear()
    _tool_stats.max_duration.clear()
    _tool_stats.last_updated.clear()
    yield
    # 测试后再清空一次，确保不会污染其他测试
    _tool_stats.call_counts.clear()
    _tool_stats.error_counts.clear()
    _tool_stats.slow_counts.clear()
    _tool_stats.total_duration.clear()
    _tool_stats.max_duration.clear()
    _tool_stats.last_updated.clear()


class TestToolStats:
    """测试工具统计收集器"""

    def test_record_call_success(self):
        """测试记录成功调用"""
        _tool_stats.record_call("test_tool", duration=1.0, success=True)

        stats = _tool_stats.get_stats("test_tool")
        assert stats["total_calls"] == 1
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration"] == 1.0

    def test_record_call_error(self):
        """测试记录错误调用"""
        _tool_stats.record_call("test_tool", duration=0.5, success=False)

        stats = _tool_stats.get_stats("test_tool")
        assert stats["total_calls"] == 1
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 0.0

    def test_record_call_slow(self):
        """测试记录慢调用"""
        _tool_stats.record_call("test_tool", duration=6.0, success=True, slow_threshold=5.0)

        stats = _tool_stats.get_stats("test_tool")
        assert stats["slow_count"] == 1
        assert stats["slow_rate"] == 1.0

    def test_multiple_calls_aggregation(self):
        """测试多次调用聚合"""
        _tool_stats.record_call("test_tool", duration=1.0, success=True)
        _tool_stats.record_call("test_tool", duration=2.0, success=True)
        _tool_stats.record_call("test_tool", duration=0.5, success=False)

        stats = _tool_stats.get_stats("test_tool")
        assert stats["total_calls"] == 3
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["avg_duration"] == pytest.approx(3.5 / 3)

    def test_max_duration_tracking(self):
        """测试最大时长跟踪"""
        _tool_stats.record_call("test_tool", duration=1.0, success=True)
        _tool_stats.record_call("test_tool", duration=5.0, success=True)
        _tool_stats.record_call("test_tool", duration=2.0, success=True)

        stats = _tool_stats.get_stats("test_tool")
        assert stats["max_duration"] == 5.0

    def test_get_all_stats(self):
        """测试获取所有工具统计"""
        _tool_stats.record_call("tool_a", duration=1.0, success=True)
        _tool_stats.record_call("tool_b", duration=2.0, success=True)

        all_stats = _tool_stats.get_all_stats()
        tool_names = [s["tool_name"] for s in all_stats]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names


class TestMonitorToolExecution:
    """测试工具执行监控上下文管理器"""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """测试成功执行监控"""
        tool_name = "test_tool"

        async with monitor_tool_execution(tool_name):
            await asyncio.sleep(0.01)

        # 验证统计已更新
        stats = _tool_stats.get_stats(tool_name)
        assert stats["total_calls"] == 1
        assert stats["error_count"] == 0

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """测试失败执行监控"""
        tool_name = "test_tool"

        with pytest.raises(ValueError):
            async with monitor_tool_execution(tool_name):
                raise ValueError("测试错误")

        # 验证错误已记录
        stats = _tool_stats.get_stats(tool_name)
        assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_duration_recording(self):
        """测试时长记录"""
        tool_name = "test_tool"

        async with monitor_tool_execution(tool_name):
            time.sleep(0.01)

        stats = _tool_stats.get_stats(tool_name)
        assert stats["avg_duration"] > 0


class TestMonitoredTool:
    """测试监控工具包装器"""

    @pytest.mark.asyncio
    async def test_wrap_tool(self):
        """测试包装工具"""
        @tool
        async def original_tool(query: str) -> str:
            """原始工具"""
            return f"结果: {query}"

        monitored = MonitoredTool.wrap(original_tool)

        assert monitored.name == "original_tool"
        # 使用 object.__getattribute__ 访问私有属性
        assert object.__getattribute__(monitored, "_original_tool") == original_tool

    @pytest.mark.asyncio
    async def test_execute_monitored(self):
        """测试执行监控工具"""
        @tool
        async def original_tool(query: str) -> str:
            """原始工具"""
            return f"结果: {query}"

        monitored = MonitoredTool.wrap(original_tool)
        # LangChain 的 _arun 需要 config 参数
        result = await monitored._arun("测试", config={})

        assert result == "结果: 测试"

        # 验证监控已记录
        stats = _tool_stats.get_stats("original_tool")
        assert stats["total_calls"] == 1

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """测试错误传播"""
        @tool
        async def failing_tool() -> str:
            """失败工具"""
            raise ValueError("工具错误")

        monitored = MonitoredTool.wrap(failing_tool)

        with pytest.raises(ValueError, match="工具错误"):
            # LangChain 的 _arun 需要 config 参数
            await monitored._arun(config={})

        # 验证错误已记录
        stats = _tool_stats.get_stats("failing_tool")
        assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_custom_slow_threshold(self):
        """测试自定义慢调用阈值"""
        @tool
        async def slow_tool() -> str:
            """慢工具"""
            await asyncio.sleep(0.1)
            return "done"

        monitored = MonitoredTool.wrap(slow_tool, slow_threshold=0.05)
        await monitored._arun(config={})

        stats = _tool_stats.get_stats("slow_tool")
        assert stats["slow_count"] == 1


class TestWrapToolsWithMonitoring:
    """测试批量包装工具"""

    def test_wrap_empty_list(self):
        """测试包装空列表"""
        result = wrap_tools_with_monitoring([])
        assert result == []

    def test_wrap_single_tool(self):
        """测试包装单个工具"""
        @tool
        def test_tool() -> str:
            return "test"

        result = wrap_tools_with_monitoring([test_tool])
        assert len(result) == 1
        assert isinstance(result[0], MonitoredTool)

    def test_wrap_multiple_tools(self):
        """测试包装多个工具"""
        @tool
        def tool_a() -> str:
            return "a"

        @tool
        def tool_b() -> str:
            return "b"

        result = wrap_tools_with_monitoring([tool_a, tool_b])
        assert len(result) == 2
        assert all(isinstance(t, MonitoredTool) for t in result)

    def test_skip_already_monitored(self):
        """测试跳过已监控的工具"""
        @tool
        def test_tool() -> str:
            return "test"

        already_monitored = MonitoredTool.wrap(test_tool)
        result = wrap_tools_with_monitoring([already_monitored])

        # 应该返回同一个实例
        assert result[0] is already_monitored


class TestHealthReport:
    """测试健康度报告"""

    def test_get_single_tool_report(self):
        """测试获取单个工具报告"""
        _tool_stats.record_call("test_tool", duration=1.0, success=True)
        _tool_stats.record_call("test_tool", duration=0.5, success=False)

        report = get_tool_health_report("test_tool")

        assert report["tool_name"] == "test_tool"
        assert report["total_calls"] == 2
        assert report["success_rate"] == 0.5

    def test_get_all_tools_report(self):
        """测试获取所有工具报告"""
        _tool_stats.record_call("tool_a", duration=1.0, success=True)
        _tool_stats.record_call("tool_b", duration=2.0, success=True)

        report = get_tool_health_report()

        assert "tools" in report
        assert len(report["tools"]) == 2
        assert report["summary"]["total_tools"] == 2

    def test_format_health_report_single(self):
        """测试格式化单工具报告"""
        _tool_stats.record_call("test_tool", duration=1.0, success=True)

        report = get_tool_health_report("test_tool")
        formatted = format_health_report(report)

        assert "test_tool" in formatted
        assert "总调用" in formatted

    def test_format_health_report_multiple(self):
        """测试格式化多工具报告"""
        _tool_stats.record_call("tool_a", duration=1.0, success=True)
        _tool_stats.record_call("tool_b", duration=2.0, success=True)

        report = get_tool_health_report()
        formatted = format_health_report(report)

        assert "工具健康度报告" in formatted
        assert "tool_a" in formatted
        assert "tool_b" in formatted


class TestPrometheusMetrics:
    """测试 Prometheus 指标"""

    def test_tool_errors_total(self):
        """测试工具错误计数器"""
        assert isinstance(tool_errors_total, Counter)
        # 标签应该包含 tool_name 和 error_type（prometheus_client 返回元组）
        assert set(tool_errors_total._labelnames) == {"tool_name", "error_type"}

    def test_tool_slow_calls_total(self):
        """测试慢调用计数器"""
        assert isinstance(tool_slow_calls_total, Counter)
        assert set(tool_slow_calls_total._labelnames) == {"tool_name", "threshold_seconds"}

    def test_tool_active_gauge(self):
        """测试活跃工具 Gauge"""
        assert isinstance(tool_active_gauge, Gauge)
        assert set(tool_active_gauge._labelnames) == {"tool_name"}

    def test_tool_health_score(self):
        """测试健康度评分 Gauge"""
        assert isinstance(tool_health_score, Gauge)
        assert set(tool_health_score._labelnames) == {"tool_name"}


# 导入 asyncio 用于测试
import asyncio
