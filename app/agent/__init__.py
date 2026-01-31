"""Agent 核心模块

包含：
- state: Agent 状态定义（使用 add_messages reducer）
- graphs: LangGraph 工作流（BaseGraph + ChatGraph + ReactGraph）
- tools: 工具系统（registry + builtin 示例工具）
- tool_interceptor: 工具执行拦截器
- agent: Agent 管理类（LangGraphAgent 门面）
- multi_agent: 多 Agent 系统（Router/Supervisor/Swarm）
- factory: Agent 工厂模式（统一创建接口）
- callbacks: Callback Handler（Langfuse + Prometheus）
- prompts: Prompt 模板管理
- memory: Memory 管理（短期 + 长期 + 窗口记忆）
- retry: 工具重试机制（指数退避 + 可配置策略）

目录结构:
    agent/
    ├── __init__.py       # 本文件
    ├── state.py          # AgentState 定义（增强版，参考 DeerFlow）
    ├── retry.py          # 重试机制（参考 AI 训练营 p13-toolRetry.py）
    ├── graphs/           # 图工作流
    │   ├── base.py       # BaseGraph 抽象基类
    │   ├── chat.py       # ChatGraph 基础对话图
    │   ├── react.py      # ReactGraph ReAct 模式
    │   ├── nodes.py      # 节点函数
    │   └── routes.py     # 路由函数
    ├── tools/            # 工具系统
    │   ├── registry.py   # 工具注册表
    │   └── builtin/      # 内置示例工具
    ├── tool_interceptor.py  # 工具拦截器（参考 DeerFlow）
    ├── agent.py          # LangGraphAgent 门面类
    ├── multi_agent.py    # 多 Agent 系统
    ├── factory.py        # Agent 工厂（增强版，参考 DeerFlow）
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
from app.agent.capabilities.clarification import (
    ClarificationNode,
    build_clarified_query,
    build_clarified_topic_from_history,
    complete_clarification,
    create_clarification_prompt,
    format_clarification_context,
    get_clarification_summary,
    needs_clarification,
    record_clarification,
    reset_clarification,
    should_prompt_clarification,
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
from app.agent.graphs import (
    BaseGraph,
    ChatGraph,
    ReactAgent,
    chat_node,
    create_chat_graph,
    create_react_agent,
    route_by_tools,
    tools_node,
)
from app.agent.multi_agent import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_handoff_tool,
    create_multi_agent_system,
    create_swarm,
)
from app.agent.state import (
    AgentState,
    create_initial_state,
    create_state_from_input,
    get_default_state,
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

# 直接导出 streaming 主要类和函数
from app.agent.streaming.streaming import (  # noqa: E402, F401
    DoneEvent,
    ErrorEvent,
    StatusEvent,
    StreamEvent,
    StreamingAgent,
    TokenEvent,
    TokenStream,
    collect_stream,
    create_fastapi_streaming_response,
    stream_agent_response,
    stream_agent_sse,
)

# 上下文管理和 RAG
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
from app.agent.capabilities.rag import (  # noqa: E402, F401
    BaseVectorStore,
    ChromaStore,
    PgVectorStore,
    PineconeStore,
    SearchResult,
    VectorStoreConfig,
    VectorStoreType,
    create_vector_store,
    index_documents,
    retrieve_documents,
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
    # State
    "AgentState",
    "create_initial_state",
    "create_state_from_input",
    "get_default_state",
    # Graphs - 抽象基类
    "BaseGraph",
    # Graphs - 具体实现
    "ChatGraph",
    "ReactAgent",
    "create_chat_graph",
    "create_react_agent",
    # Graphs - 节点和路由（供扩展使用）
    "chat_node",
    "tools_node",
    "route_by_tools",
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
    # Multi-Agent
    "RouterAgent",
    "SupervisorAgent",
    "HandoffAgent",
    "create_handoff_tool",
    "create_swarm",
    "create_multi_agent_system",
    # Clarification - 意图澄清
    "needs_clarification",
    "should_prompt_clarification",
    "record_clarification",
    "complete_clarification",
    "reset_clarification",
    "build_clarified_query",
    "build_clarified_topic_from_history",
    "get_clarification_summary",
    "create_clarification_prompt",
    "format_clarification_context",
    "ClarificationNode",
    # Streaming - 流式输出
    "StreamEvent",
    "TokenEvent",
    "ToolCallEvent",
    "ErrorEvent",
    "StatusEvent",
    "DoneEvent",
    "TokenStream",
    "StreamingAgent",
    "stream_agent_response",
    "stream_agent_sse",
    "collect_stream",
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
    # RAG - 检索增强
    "BaseVectorStore",
    "PgVectorStore",
    "PineconeStore",
    "ChromaStore",
    "VectorStoreConfig",
    "VectorStoreType",
    "SearchResult",
    "create_vector_store",
    "retrieve_documents",
    "index_documents",
    # Memory - 窗口记忆
    "TrimStrategy",
    "TokenCounterType",
    "WindowMemoryManager",
    "create_pre_model_hook",
    "create_chat_hook",
    "get_window_memory_manager",
    "trim_state_messages",
]

# 向后兼容：保留旧名称
AgentGraph = ChatGraph  # type: ignore
create_agent_graph = create_chat_graph  # type: ignore
