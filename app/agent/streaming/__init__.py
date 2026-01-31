"""LangGraph 流式处理模块

提供流式处理、取消和继续支持。
"""

# 从原始 streaming.py 模块导入
from app.agent.streaming.base import (
    StreamEvent,
    StreamProcessor,
    stream_events_from_graph,
    stream_tokens_from_graph,
)

# 从 cancellation 模块导入取消相关功能
from app.agent.streaming.cancellation import (
    CancellableStreamProcessor,
    CancellationToken,
    SessionStoppedError,
    is_session_stopped,
    request_session_stop,
    reset_session_stop,
)

# 从 continuation 模块导入继续功能
from app.agent.streaming.continuation import (
    ActiveStreamRegistry,
    ContinuableStreamProcessor,
    get_active_stream_registry,
)

__all__ = [
    # 原始流式处理
    "StreamEvent",
    "StreamProcessor",
    "stream_tokens_from_graph",
    "stream_events_from_graph",
    # 取消功能
    "CancellableStreamProcessor",
    "CancellationToken",
    "SessionStoppedError",
    "request_session_stop",
    "is_session_stopped",
    "reset_session_stop",
    # 继续功能
    "ActiveStreamRegistry",
    "ContinuableStreamProcessor",
    "get_active_stream_registry",
]
