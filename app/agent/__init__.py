"""Agent 核心模块（DeerFlow 风格）

架构原则：
- 所有 Agent 都是 CompiledStateGraph
- 不需要额外的类包装
- 使用统一的 Agent 工厂创建

核心模块：
- graph: LangGraph 工作流（Agent 工厂、Multi-Agent、Checkpoint 持久化）
- tools: 工具系统
- retry: 重试机制
- memory: 记忆管理
- streaming: 流式输出

使用示例：
    ```python
    from app.agent import create_agent, AGENT_REGISTRY

    # 创建 Planner
    planner = create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=[],
        prompt_template="planner",
    )

    # 调用 Agent
    result = await planner.ainvoke(
        {"messages": [("user", "创建排序算法")]},
        {"configurable": {"thread_id": "session-123"}},
    )
    ```
"""

# ============== Agent 工厂（DeerFlow 风格）=============
# ============== 上下文管理 ==============
from app.agent.context import (
    ContextManager,
    SlidingContextWindow,
    compress_context,
    count_messages_tokens,
    count_tokens,
    truncate_messages,
    truncate_text,
)

# ============== 图构建函数 ==============
# ============== Checkpoint 持久化 ==============
# ============== Human-in-the-Loop ==============
# ============== 图缓存 ==============
# ============== 工具函数 ==============
from app.agent.graph import (
    # Agent 配置
    AGENT_REGISTRY,
    CHAT_TEAM,
    DEFAULT_SYSTEM_PROMPT,
    # Agent 组
    RESEARCH_TEAM,
    SUPERVISOR_TEAM,
    # 高级图构建器（三层记忆架构）
    AdvancedGenerationBuilder,
    GraphCache,
    HumanApproval,
    InterruptGraph,
    InterruptRequest,
    # Multi-Agent
    MultiAgentGraphBuilder,
    agent_execution_context,
    build_chat_graph,
    build_multi_agent_graph,
    chat_node,
    check_interrupt_node,
    clear_graph_cache,
    close_postgres_checkpointer,
    compile_chat_graph,
    compile_interrupt_graph,
    create_advanced_generation_workflow,
    # Agent 工厂
    create_agent,
    create_interrupt_graph,
    # ReAct Agent（便捷函数）
    create_react_agent,
    create_worker_node,
    execute_node,
    get_agent_config,
    get_cached_graph,
    get_checkpoint_count,
    get_checkpointer,
    get_graph_cache,
    get_graph_cache_stats,
    get_message_content,
    get_postgres_checkpointer,
    has_tool_calls,
    interrupt_chat_node,
    invoke_chat_graph,
    is_user_message,
    list_agents,
    list_agents_by_type,
    list_checkpoints,
    run_advanced_generation,
    should_continue,
    stream_chat_graph,
    supervisor_node,
    validate_state,
)
from app.agent.message_utils import (
    extract_ai_content,
    format_messages_to_dict,
)

# ============== 重试机制 ==============
from app.agent.retry import (
    NetworkError,
    RateLimitError,
    ResourceUnavailableError,
    RetryableError,
    RetryPolicy,
    RetryStrategy,
    TemporaryServiceError,
    ToolExecutionError,
    get_default_retry_policy,
    with_retry,
)

# ============== 状态类型 ==============
from app.agent.state import (
    AdvancedGenerationState,
    AgentState,
    ChatState,
    MultiAgentState,
    ReActState,
    add_messages,
    create_agent_state,
    create_chat_state,
    create_react_state,
    create_state_from_input,
    should_stop_iteration,
)

# ============== 流式输出 ==============
from app.agent.streaming import (
    StreamEvent,
    StreamProcessor,
    stream_events_from_graph,
    stream_tokens_from_graph,
)

# ============== 工具系统 ==============
from app.agent.tools import (
    aget_tool_node,
    alist_tools,
    calculate,
    get_tool,
    get_weather,
    list_tools,
    register_tool,
    search_database,
    search_web,
)

# ============== Workflow 编排 ==============
from app.agent.workflow import graph, run_agent_workflow

# 工具拦截器（可选）
try:
    from app.agent.tools.interceptor import (
        ToolInterceptor,
        wrap_tools_with_interceptor,
    )
except ImportError:
    ToolInterceptor = None  # type: ignore
    wrap_tools_with_interceptor = None  # type: ignore

# 状态工具函数（可选）
try:
    from app.agent.state.utils import increment_iteration, preserve_state_meta_fields
except ImportError:
    increment_iteration = None  # type: ignore
    preserve_state_meta_fields = None  # type: ignore

__all__ = [
    # ============== Agent 工厂（DeerFlow 风格）=============
    "create_agent",
    "AGENT_REGISTRY",
    "get_agent_config",
    "list_agents",
    "list_agents_by_type",
    "RESEARCH_TEAM",
    "CHAT_TEAM",
    "SUPERVISOR_TEAM",
    "create_react_agent",
    # ============== Workflow 编排 ==============
    "graph",
    "run_agent_workflow",
    # ============== 状态类型 ==============
    "AgentState",
    "ChatState",
    "ReActState",
    "MultiAgentState",
    "AdvancedGenerationState",
    "add_messages",
    "create_agent_state",
    "create_chat_state",
    "create_react_state",
    "create_state_from_input",
    "should_stop_iteration",
    "increment_iteration",
    "preserve_state_meta_fields",
    # ============== 图构建 ==============
    "DEFAULT_SYSTEM_PROMPT",
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    "chat_node",
    # ============== Multi-Agent ==============
    "MultiAgentGraphBuilder",
    "build_multi_agent_graph",
    "create_worker_node",
    "supervisor_node",
    "agent_execution_context",
    # ============== 高级图构建器（三层记忆架构）=============
    "AdvancedGenerationBuilder",
    "create_advanced_generation_workflow",
    "run_advanced_generation",
    # ============== Checkpoint 持久化 ==============
    "get_postgres_checkpointer",
    "get_checkpointer",
    "close_postgres_checkpointer",
    "list_checkpoints",
    "get_checkpoint_count",
    # ============== Human-in-the-Loop ==============
    "InterruptGraph",
    "create_interrupt_graph",
    "HumanApproval",
    "InterruptRequest",
    "check_interrupt_node",
    "compile_interrupt_graph",
    "interrupt_chat_node",
    "execute_node",
    # ============== 工具 ==============
    "register_tool",
    "get_tool",
    "list_tools",
    "aget_tool_node",
    "alist_tools",
    "calculate",
    "get_weather",
    "search_database",
    "search_web",
    # ============== 重试 ==============
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
    # ============== 流式输出 ==============
    "StreamEvent",
    "StreamProcessor",
    "stream_tokens_from_graph",
    "stream_events_from_graph",
    # ============== 上下文 ==============
    "ContextManager",
    "SlidingContextWindow",
    "compress_context",
    "count_tokens",
    "count_messages_tokens",
    "truncate_messages",
    "truncate_text",
    # ============== 图缓存 ==============
    "GraphCache",
    "get_graph_cache",
    "get_cached_graph",
    "clear_graph_cache",
    "get_graph_cache_stats",
    # ============== 工具函数 ==============
    "get_message_content",
    "is_user_message",
    "format_messages_to_dict",
    "extract_ai_content",
    "validate_state",
    "has_tool_calls",
    "should_continue",
    # ============== 工具拦截器 ==============
    "ToolInterceptor",
    "wrap_tools_with_interceptor",
]
