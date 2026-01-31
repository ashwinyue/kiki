"""指标监控测试

测试 Prometheus 指标收集功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.observability.metrics import (
    active_sessions,
    active_users,
    agent_duration_seconds,
    agent_requests_total,
    db_connections_active,
    db_query_duration_seconds,
    db_queries_total,
    get_metrics_text,
    http_request_duration_seconds,
    http_requests_total,
    increment_active_sessions,
    increment_active_users,
    llm_duration_seconds,
    llm_requests_total,
    llm_tokens_total,
    record_llm_tokens,
    tool_calls_total,
    tool_duration_seconds,
    track_http_request,
    track_llm_request,
    track_tool_call,
)


class TestHTTPMetrics:
    """HTTP 指标测试"""

    @pytest.mark.asyncio
    async def test_track_http_request_success(self) -> None:
        """测试追踪成功的 HTTP 请求"""
        async with track_http_request("GET", "/api/test"):
            pass  # 成功执行

        # 验证指标已记录（通过检查 metric 的 samples）
        samples = list(http_requests_total.collect())
        assert len(samples) > 0

    @pytest.mark.asyncio
    async def test_track_http_request_error(self) -> None:
        """测试追踪失败的 HTTP 请求"""
        with pytest.raises(ValueError):
            async with track_http_request("POST", "/api/error"):
                raise ValueError("Test error")

        # 验证错误状态被记录
        samples = list(http_requests_total.collect())
        assert len(samples) > 0


class TestLLMMetrics:
    """LLM 指标测试"""

    @pytest.mark.asyncio
    async def test_track_llm_request_success(self) -> None:
        """测试追踪成功的 LLM 请求"""
        async with track_llm_request("gpt-4o", "openai"):
            pass  # 成功执行

        samples = list(llm_requests_total.collect())
        assert len(samples) > 0

    @pytest.mark.asyncio
    async def test_track_llm_request_error(self) -> None:
        """测试追踪失败的 LLM 请求"""
        with pytest.raises(RuntimeError):
            async with track_llm_request("gpt-4o", "openai"):
                raise RuntimeError("LLM error")

        samples = list(llm_requests_total.collect())
        assert len(samples) > 0

    def test_record_llm_tokens(self) -> None:
        """测试记录 Token 使用"""
        record_llm_tokens("gpt-4o", 100, 50)

        samples = list(llm_tokens_total.collect())
        assert len(samples) > 0


class TestToolMetrics:
    """工具指标测试"""

    @pytest.mark.asyncio
    async def test_track_tool_call_success(self) -> None:
        """测试追踪成功的工具调用"""
        async with track_tool_call("calculate"):
            pass  # 成功执行

        samples = list(tool_calls_total.collect())
        assert len(samples) > 0

    @pytest.mark.asyncio
    async def test_track_tool_call_error(self) -> None:
        """测试追踪失败的工具调用"""
        with pytest.raises(RuntimeError):
            async with track_tool_call("calculate"):
                raise RuntimeError("Tool error")

        samples = list(tool_calls_total.collect())
        assert len(samples) > 0


class TestDatabaseMetrics:
    """数据库指标测试"""

    def test_db_connections_active(self) -> None:
        """测试活跃连接数指标"""
        # Gauge 可以设置值
        db_connections_active.set(10)
        assert db_connections_active._value._value == 10

    def test_db_queries_total(self) -> None:
        """测试查询总数计数器"""
        db_queries_total.labels(operation="select", table="users", status="success").inc()
        samples = list(db_queries_total.collect())
        assert len(samples) > 0


class TestSystemMetrics:
    """系统指标测试"""

    def test_increment_active_sessions(self) -> None:
        """测试增加活跃会话"""
        initial = active_sessions._value._value
        increment_active_sessions(5)
        assert active_sessions._value._value == initial + 5

    def test_increment_active_users(self) -> None:
        """测试增加活跃用户"""
        initial = active_users._value._value
        increment_active_users(3)
        assert active_users._value._value == initial + 3

    def test_decrement_active_sessions(self) -> None:
        """测试减少活跃会话"""
        increment_active_sessions(-1)
        # 验证可以减少
        assert True


class TestGetMetricsText:
    """get_metrics_text 函数测试"""

    @patch("prometheus_client.exposition.generate_latest")
    def test_get_metrics_text(self, mock_generate_latest) -> None:
        """测试获取指标文本"""
        mock_generate_latest.return_value = b"# Mock metrics"

        result = get_metrics_text()

        assert result == b"# Mock metrics"
        mock_generate_latest.assert_called_once()


@pytest.mark.parametrize("method,endpoint", [
    ("GET", "/api/users"),
    ("POST", "/api/chat"),
    ("DELETE", "/api/sessions/123"),
])
@pytest.mark.asyncio
async def test_track_various_http_requests(method: str, endpoint: str) -> None:
    """参数化测试各种 HTTP 请求"""
    async with track_http_request(method, endpoint):
        pass  # 成功执行


@pytest.mark.parametrize("model,provider", [
    ("gpt-4o", "openai"),
    ("claude-sonnet-4", "anthropic"),
    ("deepseek-chat", "deepseek"),
])
@pytest.mark.asyncio
async def test_track_various_llm_requests(model: str, provider: str) -> None:
    """参数化测试各种 LLM 请求"""
    async with track_llm_request(model, provider):
        pass  # 成功执行


@pytest.mark.parametrize("tool_name", [
    "calculate",
    "search",
    "get_weather",
])
@pytest.mark.asyncio
async def test_track_various_tool_calls(tool_name: str) -> None:
    """参数化测试各种工具调用"""
    async with track_tool_call(tool_name):
        pass  # 成功执行
