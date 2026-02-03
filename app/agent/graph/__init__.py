"""LangGraph 工作流模块

提供 LangGraph 工作流构建和执行功能。

使用示例:
    ```python
    from app.agent.graph import compile_chat_graph, invoke_chat_graph

    # 编译图后调用
    graph = compile_chat_graph()
    result = await graph.ainvoke(
        {"messages": [("user", "你好")]},
        {"configurable": {"thread_id": "session-123"}}
    )
    ```
"""

# 状态类型
# 构建函数
# 高级图构建器（三层记忆架构）
from app.agent.graph.advanced_builder import (
    AdvancedGenerationBuilder,
    create_advanced_generation_workflow,
    run_advanced_generation,
)

# Agent 工厂（参考 DeerFlow 设计）
from app.agent.graph.agent_factory import create_agent

# Agent 配置
from app.agent.graph.agents import (
    AGENT_REGISTRY,
    CHAT_TEAM,
    RESEARCH_TEAM,
    SUPERVISOR_TEAM,
    get_agent_config,
    list_agents,
    list_agents_by_type,
)
from app.agent.graph.builder import (
    build_chat_graph,
    compile_chat_graph,
    invoke_chat_graph,
    stream_chat_graph,
)

# 图缓存
from app.agent.graph.cache import (
    GraphCache,
    clear_graph_cache,
    get_cached_graph,
    get_graph_cache,
    get_graph_cache_stats,
)

# Checkpoint 持久化
from app.agent.graph.checkpoint import (
    close_postgres_checkpointer,
    get_checkpoint_count,
    get_checkpointer,
    get_postgres_checkpointer,
    list_checkpoints,
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

# Multi-Agent
from app.agent.graph.multi_agent import (
    MultiAgentGraphBuilder,
    agent_execution_context,
    build_multi_agent_graph,
    create_worker_node,
    supervisor_node,
)

# 节点函数
from app.agent.graph.nodes import (
    chat_node,
)

# ReAct Agent
from app.agent.graph.react import (
    ReactAgent,
    create_react_agent,
)

# 所有 State 类从 state.py 导入（避免重复定义）
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
    # 注意: increment_iteration, preserve_state_meta_fields 不在 state.__all__ 中
    # 如果需要这些函数，从 types.py 或其他地方导入
    should_stop_iteration,
)

# 其他工具函数从 state.utils 导入
try:
    from app.agent.state.utils import increment_iteration
except ImportError:
    pass
try:
    from app.agent.state.utils import preserve_state_meta_fields
except ImportError:
    pass

# 工具函数
from app.agent.graph.utils import (
    get_message_content,
    has_tool_calls,
    is_user_message,
    should_continue,
    validate_state,
)
from app.agent.message_utils import (
    extract_ai_content,
    format_messages_to_dict,
)

__all__ = [
    # ============== 状态类型 ==============
    "ChatState",
    "AgentState",
    "ReActState",
    "MultiAgentState",
    "AdvancedGenerationState",
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
    # ============== 构建函数 ==============
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    # ============== Checkpoint 持久化 ==============
    "get_postgres_checkpointer",
    "get_checkpointer",
    "close_postgres_checkpointer",
    "list_checkpoints",
    "get_checkpoint_count",
    # ============== Multi-Agent ==============
    "MultiAgentGraphBuilder",
    "build_multi_agent_graph",
    "create_worker_node",
    "supervisor_node",
    "agent_execution_context",
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
    # ============== Agent 工厂 ==============
    "create_agent",
    # ============== Agent 配置 ==============
    "AGENT_REGISTRY",
    "get_agent_config",
    "list_agents",
    "list_agents_by_type",
    "RESEARCH_TEAM",
    "SUPERVISOR_TEAM",
    "CHAT_TEAM",
    # ============== 高级图构建器（三层记忆架构）=============
    "AdvancedGenerationBuilder",
    "create_advanced_generation_workflow",
    "run_advanced_generation",
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
