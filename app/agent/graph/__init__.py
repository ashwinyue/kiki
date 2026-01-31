"""LangGraph 工作流模块

参考 DeerFlow 项目结构，提供 LangGraph 工作流构建和执行功能。

目录结构:
    graph/
    ├── types.py      # 状态定义
    ├── nodes.py      # 节点函数
    ├── builder.py    # 图构建函数
    ├── utils.py      # 工具函数
    ├── interrupt.py  # Human-in-the-Loop
    ├── react.py      # ReAct Agent
    └── cache.py      # 图缓存

使用示例:
    ```python
    from app.agent.graph import compile_chat_graph, invoke_chat_graph

    # 方式1: 编译图后调用
    graph = compile_chat_graph()
    result = await graph.ainvoke(
        {"messages": [("user", "你好")]},
        {"configurable": {"thread_id": "session-123"}}
    )

    # 方式2: 使用便捷函数
    result = await invoke_chat_graph(
        message="你好",
        session_id="session-123"
    )

    # 方式3: 使用 InterruptGraph
    from app.agent.graph import create_interrupt_graph

    graph = create_interrupt_graph()
    await graph.ainvoke(...)
    ```
"""

# 状态类型
from app.agent.graph.types import (
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
)

# 节点函数
from app.agent.graph.nodes import (
    chat_node,
    create_chat_node_factory,
    route_by_tools,
    tools_node,
)

# 构建函数
from app.agent.graph.builder import (
    build_chat_graph,
    compile_chat_graph,
    invoke_chat_graph,
    stream_chat_graph,
)

# Human-in-the-Loop
from app.agent.graph.interrupt import (
    HumanApproval,
    InterruptGraph,
    InterruptRequest,
    check_interrupt_node,
    compile_interrupt_graph,
    create_interrupt_graph,
    execute_node,
    interrupt_chat_node,
)

# ReAct Agent
from app.agent.graph.react import (
    ReactAgent,
    create_react_agent,
)

# 图缓存
from app.agent.graph.cache import (
    GraphCache,
    clear_graph_cache,
    get_cached_graph,
    get_graph_cache,
    get_graph_cache_stats,
)

# 工具函数
from app.agent.graph.utils import (
    extract_ai_content,
    format_messages_to_dict,
    get_message_content,
    has_tool_calls,
    is_user_message,
    should_continue,
    validate_state,
)

__all__ = [
    # ============== 状态类型 ==============
    "ChatState",
    "AgentState",
    "ReActState",
    "add_messages",
    "create_chat_state",
    "create_agent_state",
    "create_react_state",
    "create_state_from_input",
    "increment_iteration",
    "should_stop_iteration",
    "preserve_state_meta_fields",
    # ============== 节点函数 ==============
    "chat_node",
    "tools_node",
    "route_by_tools",
    "create_chat_node_factory",
    # ============== 构建函数 ==============
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    # ============== Human-in-the-Loop ==============
    "InterruptGraph",
    "create_interrupt_graph",
    "HumanApproval",
    "InterruptRequest",
    "check_interrupt_node",
    "compile_interrupt_graph",
    "interrupt_chat_node",
    "execute_node",
    # ============== ReAct Agent ==============
    "ReactAgent",
    "create_react_agent",
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
]
