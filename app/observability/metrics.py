"""指标监控模块

使用 Prometheus 进行应用指标收集。
"""

import time
from contextlib import asynccontextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from app.observability.logging import get_logger

logger = get_logger(__name__)

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=registry,
)

agent_requests_total = Counter(
    "agent_requests_total",
    "Agent 请求总数",
    ["agent_type", "status"],
    registry=registry,
)

agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Agent 处理耗时（秒）",
    ["agent_type"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

llm_requests_total = Counter(
    "llm_requests_total",
    "LLM 请求总数",
    ["model", "provider", "status"],
    registry=registry,
)

llm_duration_seconds = Histogram(
    "llm_duration_seconds",
    "LLM 推理耗时（秒）",
    ["model", "provider"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    registry=registry,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM Token 使用总数",
    ["model", "token_type"],  # token_type: prompt/completion
    registry=registry,
)

tool_calls_total = Counter(
    "tool_calls_total",
    "工具调用总数",
    ["tool_name", "status"],
    registry=registry,
)

tool_duration_seconds = Histogram(
    "tool_duration_seconds",
    "工具调用耗时（秒）",
    ["tool_name"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry,
)

db_connections_active = Gauge(
    "db_connections_active",
    "活跃数据库连接数",
    registry=registry,
)

db_queries_total = Counter(
    "db_queries_total",
    "数据库查询总数",
    ["operation", "table", "status"],
    registry=registry,
)

db_query_duration_seconds = Histogram(
    "db_query_duration_seconds",
    "数据库查询耗时（秒）",
    ["operation", "table"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry,
)

active_sessions = Gauge(
    "active_sessions",
    "活跃会话数",
    registry=registry,
)

active_users = Gauge(
    "active_users",
    "活跃用户数",
    registry=registry,
)


@asynccontextmanager
async def track_http_request(method: str, endpoint: str):
    """追踪 HTTP 请求

    Args:
        method: HTTP 方法
        endpoint: 端点路径

    Yields:
        None
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception as e:
        status = "error"
        logger.warning("http_request_error", method=method, endpoint=endpoint, error=str(e))
        raise
    finally:
        duration = time.time() - start_time
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


@asynccontextmanager
async def track_llm_request(model: str, provider: str):
    """追踪 LLM 请求

    Args:
        model: 模型名称
        provider: 提供商名称

    Yields:
        None
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception as e:
        status = "error"
        logger.warning("llm_request_error", model=model, provider=provider, error=str(e))
        raise
    finally:
        duration = time.time() - start_time
        llm_requests_total.labels(model=model, provider=provider, status=status).inc()
        llm_duration_seconds.labels(model=model, provider=provider).observe(duration)


@asynccontextmanager
async def track_tool_call(tool_name: str):
    """追踪工具调用

    Args:
        tool_name: 工具名称

    Yields:
        None
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception as e:
        status = "error"
        logger.warning("tool_call_error", tool_name=tool_name, error=str(e))
        raise
    finally:
        duration = time.time() - start_time
        tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        tool_duration_seconds.labels(tool_name=tool_name).observe(duration)


def record_llm_tokens(model: str, prompt_tokens: int, completion_tokens: int):
    """记录 LLM Token 使用

    Args:
        model: 模型名称
        prompt_tokens: 输入 Token 数
        completion_tokens: 输出 Token 数
    """
    llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
    llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)


def increment_active_sessions(delta: int = 1):
    """增减活跃会话数

    Args:
        delta: 变化量（正数增加，负数减少）
    """
    active_sessions.inc(delta)


def increment_active_users(delta: int = 1):
    """增减活跃用户数

    Args:
        delta: 变化量（正数增加，负数减少）
    """
    active_users.inc(delta)


def get_metrics_text() -> str:
    """获取 Prometheus 文本格式的指标

    Returns:
        指标文本
    """
    from prometheus_client import exposition

    return exposition.generate_latest(registry)
