"""Agent 核心模块

包含：
- BaseAgent: Agent 抽象基类
- ChatAgent: 标准对话 Agent
- ReactAgent: ReAct 模式 Agent
- graph: LangGraph 工作流
- tools: 工具系统
- retry: 重试机制

使用示例:
```python
from app.agent import ChatAgent, ReactAgent

# Chat Agent - 标准对话
async with ChatAgent(system_prompt="...") as agent:
    response = await agent.get_response("你好", session_id="session-123")

# React Agent - 需要工具调用
async with ReactAgent(tools=[my_tool]) as agent:
    response = await agent.get_response("今天天气?", session_id="session-123")
```
"""

# ============== 核心 Agent 类（新，推荐使用）=============
from app.agent.base import BaseAgent
from app.agent.chat_agent import ChatAgent
from app.agent.multi_agent import (
    MultiAgent,
    RouterAgent,
    SupervisorAgent,
)

# ============== 上下文管理（简化导入）=============
from app.agent.context import (
    ContextManager,
    SlidingContextWindow,
    compress_context,
    count_messages_tokens,
    count_tokens,
    truncate_messages,
    truncate_text,
)

# ============== 图构建函数（只导入核心）=============
from app.agent.graph.builder import (
    DEFAULT_SYSTEM_PROMPT,
    build_chat_graph,
    compile_chat_graph,
    invoke_chat_graph,
    stream_chat_graph,
)

# ============== 节点函数 ==============
from app.agent.graph.nodes import chat_node

# ============== ReAct Agent 创建函数 ==============
from app.agent.graph.react import ReactAgent, create_react_agent

# ============== 重试机制（简化导入）=============
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

# ============== 状态类型（从统一目录导入）=============
from app.agent.state import (
    AgentState,
    ChatState,
    ReActState,
    add_messages,
    create_agent_state,
    create_chat_state,
    create_react_state,
    create_state_from_input,
)

# ============== 流式输出 ==============
from app.agent.streaming import (
    StreamEvent,
    StreamProcessor,
    stream_events_from_graph,
    stream_tokens_from_graph,
)

# ============== 工具系统（简化导入）=============
from app.agent.tools import (
    # 工具注册
    aget_tool_node,
    alist_tools,
    # 内置工具
    calculate,
    get_tool,
    get_weather,
    list_tools,
    register_tool,
    search_database,
    search_web,
)

# ============== 记忆管理（简化导入）=============
# 窗口记忆功能已移至 app.agent.context.sliding_window
# 使用: from app.agent.context import SlidingContextWindow

# ============== Human-in-the-Loop ==============
try:
    from app.agent.graph.interrupt import (
        HumanApproval,
        InterruptGraph,
        InterruptRequest,
        create_interrupt_graph,
    )
except ImportError:
    # 如果某些导入失败，创建占位符
    InterruptGraph = None  # type: ignore
    create_interrupt_graph = None  # type: ignore
    HumanApproval = None  # type: ignore
    InterruptRequest = None  # type: ignore

# ============== 工具拦截器 ==============
try:
    from app.agent.tools.interceptor import (
        ToolInterceptor,
        wrap_tools_with_interceptor,
    )
except ImportError:
    ToolInterceptor = None  # type: ignore
    wrap_tools_with_interceptor = None  # type: ignore

# ============== 图缓存 ==============
try:
    from app.agent.graph.cache import (
        GraphCache,
        clear_graph_cache,
        get_cached_graph,
    )
except ImportError:
    GraphCache = None  # type: ignore
    clear_graph_cache = None  # type: ignore
    get_cached_graph = None  # type: ignore

__all__ = [
    # ============== 核心 Agent（新，推荐使用）=============
    "BaseAgent",
    "ChatAgent",
    "ReactAgent",
    "MultiAgent",
    "SupervisorAgent",
    "RouterAgent",
    # ============== 状态类型 ==============
    "AgentState",
    "ChatState",
    "ReActState",
    "add_messages",
    "create_agent_state",
    "create_chat_state",
    "create_react_state",
    "create_state_from_input",
    # ============== 图构建 ==============
    "DEFAULT_SYSTEM_PROMPT",
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    "create_react_agent",
    # ============== 节点 ==============
    "chat_node",
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
    # ============== 记忆 ==============
    # 注意: 窗口记忆功能已移至 app.agent.context.sliding_window
    # ============== Human-in-the-Loop ==============
    "InterruptGraph",
    "create_interrupt_graph",
    "HumanApproval",
    "InterruptRequest",
    # ============== 工具拦截器 ==============
    "ToolInterceptor",
    "wrap_tools_with_interceptor",
    # ============== 图缓存 ==============
    "GraphCache",
    "clear_graph_cache",
    "get_cached_graph",
]
