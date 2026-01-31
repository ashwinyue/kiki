"""Agent 核心模块

包含：
- graph: LangGraph 工作流（参考 DeerFlow 结构）
- tools: 工具系统（registry + builtin 示例工具）
- agent: Agent 管理类（LangGraphAgent 门面）
- factory: Agent 工厂模式（统一创建接口）
- callbacks: Callback Handler（Langfuse + Prometheus）
- prompts: Prompt 模板管理
- memory: Memory 管理（短期 + 长期 + 窗口记忆）
- retry: 工具重试机制（指数退避 + 可配置策略）

目录结构:
    agent/
    ├── __init__.py       # 本文件
    ├── graph/            # LangGraph 工作流（新结构，参考 DeerFlow）
    │   ├── types.py      # 状态定义
    │   ├── nodes.py      # 节点函数
    │   ├── builder.py    # 图构建函数
    │   └── utils.py      # 工具函数
    ├── tools/            # 工具系统
    │   ├── registry.py   # 工具注册表
    │   └── builtin/      # 内置示例工具
    ├── agent.py          # LangGraphAgent 门面类
    ├── factory.py        # Agent 工厂
    ├── callbacks/        # Callback Handlers
    ├── prompts/          # Prompt 模板
    └── memory/           # Memory 管理
"""

# 核心模块
from app.agent.agent import (
    LangGraphAgent,
    create_agent,
    get_agent,
)
from app.agent.factory import (
    AGENT_LLM_MAP,
    AgentConfig,
    AgentFactory,
    AgentFactoryError,
    AgentType,
    LLMType,
)
from app.agent.factory import (
    create_agent as factory_create_agent,
)

# 图模块（新结构，推荐使用）
from app.agent.graph import (
    # 状态类型
    AgentState,
    ChatState,
    ReActState,
    add_messages,
    create_agent_state,
    create_chat_state,
    create_react_state,
    create_state_from_input,
    increment_iteration,
    preserve_state_meta_fields,
    should_stop_iteration,
    # 构建函数
    build_chat_graph,
    compile_chat_graph,
    invoke_chat_graph,
    stream_chat_graph,
    # 节点函数
    chat_node,
    create_chat_node_factory,
    route_by_tools,
    tools_node,
    # Human-in-the-Loop
    HumanApproval,
    InterruptGraph,
    InterruptRequest,
    create_interrupt_graph,
    # ReAct Agent
    ReactAgent,
    create_react_agent,
    # 图缓存
    GraphCache,
    get_cached_graph,
    clear_graph_cache,
    get_graph_cache_stats,
    # 工具函数
    get_message_content,
    extract_ai_content,
    has_tool_calls,
    is_user_message,
)

from app.agent.tools.interceptor import (
    ToolInterceptor,
    ToolExecutionResult,
    create_tool_interceptor,
    wrap_tools_with_interceptor,
)
from app.agent.retry.retry import (  # noqa: E402, F401
    NetworkError,
    RateLimitError,
    ResourceUnavailableError,
    RetryContext,
    RetryPolicy,
    RetryStrategy,
    TemporaryServiceError,
    ToolExecutionError,
    RetryableError,
    create_retryable_node,
    execute_with_retry,
    get_default_retry_policy,
    with_retry,
)

# 流式输出模块（可选导入，避免循环依赖）
def _get_streaming():
    from app.agent import streaming

    return streaming

# 流式输出模块（基于 LangGraph）
from app.agent.streaming import (  # noqa: E402, F401
    StreamEvent,
    StreamProcessor,
    stream_events_from_graph,
    stream_tokens_from_graph,
)

# 上下文管理
from app.agent.context import (  # noqa: E402, F401
    ContextCompressor,
    ContextManager,
    SlidingContextWindow,
    compress_context,
    count_messages_tokens,
    count_tokens,
    count_tokens_precise,
    truncate_messages,
    truncate_text,
)
from app.agent.memory.window import (  # noqa: E402, F401
    TrimStrategy,
    TokenCounterType,
    WindowMemoryManager,
    create_chat_hook,
    create_pre_model_hook,
    get_window_memory_manager,
    trim_state_messages,
)
from app.agent.tools import (
    aget_tool_node,
    alist_tools,
    calculate,
    get_tool,
    get_tool_node,
    get_weather,
    list_tools,
    register_tool,
    search_database,
    search_web,
)


# 可选模块（延迟导入）
def _get_callbacks():
    from app.agent import callbacks

    return callbacks


def _get_prompts():
    from app.agent import prompts

    return prompts


def _get_memory():
    from app.agent import memory

    return memory


def _get_retry():
    from app.agent import retry

    return retry


def _get_window_memory():
    from app.agent.memory import window

    return window


__all__ = [
    # ============== 图模块（新，推荐使用）=============
    # State
    "ChatState",
    "AgentState",
    "ReActState",
    "add_messages",
    "create_chat_state",
    "create_agent_state",
    "create_react_state",
    "create_state_from_input",
    # Builder
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    # Nodes
    "chat_node",
    "tools_node",
    "route_by_tools",
    "create_chat_node_factory",
    # Utils
    "get_message_content",
    "is_user_message",
    "format_messages_to_dict",
    "extract_ai_content",
    "preserve_state_meta_fields",
    "should_stop_iteration",
    "has_tool_calls",
    # Human-in-the-Loop
    "InterruptGraph",
    "create_interrupt_graph",
    "HumanApproval",
    "InterruptRequest",
    # ReAct Agent
    "ReactAgent",
    "create_react_agent",
    # Graph Cache
    "GraphCache",
    "get_graph_cache",
    "get_cached_graph",
    "clear_graph_cache",
    "get_graph_cache_stats",
    # ============== 其他模块 ==============
    # Tools - 注册系统
    "register_tool",
    "get_tool",
    "list_tools",
    "get_tool_node",
    "alist_tools",  # 异步版本，包含 MCP 工具
    "aget_tool_node",  # 异步版本，包含 MCP 工具
    # Tools - 内置工具
    "search_web",
    "search_database",
    "get_weather",
    "calculate",
    # Tools - 拦截器
    "ToolInterceptor",
    "ToolExecutionResult",
    "wrap_tools_with_interceptor",
    "create_tool_interceptor",
    # Retry - 重试机制
    "RetryableError",
    "NetworkError",
    "RateLimitError",
    "ResourceUnavailableError",
    "TemporaryServiceError",
    "ToolExecutionError",
    "RetryStrategy",
    "RetryPolicy",
    "get_default_retry_policy",
    "with_retry",
    "RetryContext",
    "execute_with_retry",
    "create_retryable_node",
    # Agent
    "LangGraphAgent",
    "get_agent",
    "create_agent",
    # Factory
    "AgentFactory",
    "AgentFactoryError",
    "AgentType",
    "AgentConfig",
    "LLMType",
    "AGENT_LLM_MAP",
    "factory_create_agent",
    # Streaming - 流式输出（基于 LangGraph）
    "StreamEvent",
    "StreamProcessor",
    "stream_tokens_from_graph",
    "stream_events_from_graph",
    # Context - 长文本处理
    "ContextManager",
    "SlidingContextWindow",
    "ContextCompressor",
    "compress_context",
    "count_tokens",
    "count_messages_tokens",
    "count_tokens_precise",
    "truncate_messages",
    "truncate_text",
    # Memory - 窗口记忆
    "TrimStrategy",
    "TokenCounterType",
    "WindowMemoryManager",
    "create_pre_model_hook",
    "create_chat_hook",
    "get_window_memory_manager",
    "trim_state_messages",
]
