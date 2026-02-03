"""图构建函数

提供 LangGraph 工作流的构建函数。

使用 LangGraph 标准模式：
- 使用 bind_tools 绑定工具到 LLM
- 使用 ToolNode 执行工具调用
- 简化节点函数，避免重复获取工具
"""


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.agent.graph.checkpoint import get_checkpointer
from app.agent.graph.nodes import chat_node
from app.agent.state import ChatState  # 从统一状态模块导入
from app.agent.tools import alist_tools
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

# 默认最大迭代次数
DEFAULT_MAX_ITERATIONS = 10


async def create_tool_node(tenant_id: int | None = None) -> ToolNode:
    """创建工具节点（带租户工具）

    Args:
        tenant_id: 租户 ID

    Returns:
        ToolNode 实例
    """
    tools = await alist_tools(include_mcp=True, tenant_id=tenant_id)
    logger.info("tool_node_created", tool_count=len(tools))
    return ToolNode(tools)


def build_chat_graph(
    llm_service: LLMService,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> StateGraph:
    """构建聊天图（未编译）

    使用 LangGraph 标准模式：
    - 在节点外部绑定工具到 LLM
    - 使用 ToolNode 执行工具
    - 使用条件边路由

    图结构：
        START -> chat -> route_by_tools -> tools or END
        tools -> chat

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        max_iterations: 最大迭代次数

    Returns:
        StateGraph 实例（未编译）
    """
    builder = StateGraph(ChatState)

    llm_with_tools = llm_service.get_llm_with_tools()

    builder.add_node(
        "chat",
        lambda state, config: chat_node(
            state,
            config,
            llm=llm_with_tools,
            system_prompt=system_prompt,
        )
    )

    # 添加 tools 节点（将在编译时设置）
    builder.add_node("tools", lambda state, config: None)  # 占位符

    builder.add_edge(START, "chat")

    builder.add_conditional_edges(
        "chat",
        # 路由函数：检查最后一条消息是否有工具调用
        lambda state: "tools" if _has_tool_calls(state) else END,
    )

    # 工具执行后返回聊天节点
    builder.add_edge("tools", "chat")

    logger.debug("chat_graph_structure_built")
    return builder


async def compile_chat_graph(
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    tenant_id: int | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    use_postgres_checkpointer: bool = True,
) -> CompiledStateGraph:
    """编译聊天图

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器（如果提供，则不使用默认 checkpointer）
        tenant_id: 租户 ID（用于加载工具）
        max_iterations: 最大迭代次数
        use_postgres_checkpointer: 是否使用 PostgreSQL checkpointer（默认 True）

    Returns:
        编译后的 CompiledStateGraph

    Examples:
        ```python
        from app.agent.graph.builder import compile_chat_graph

        # 使用默认 PostgreSQL checkpointer
        graph = await compile_chat_graph()
        result = await graph.ainvoke(
            {"messages": [("user", "你好")]},
            {"configurable": {"thread_id": "session-123"}}
        )

        # 使用自定义 checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        custom_checkpointer = MemorySaver()
        graph = await compile_chat_graph(checkpointer=custom_checkpointer)
        ```
    """
    llm_service = llm_service or get_llm_service()
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # 构建图
    builder = build_chat_graph(llm_service, system_prompt, max_iterations)

    # 创建工具节点并替换
    tool_node = await create_tool_node(tenant_id)
    builder.add_node("tools", tool_node)

    # 获取 checkpointer
    if checkpointer is None:
        checkpointer = await get_checkpointer(use_postgres=use_postgres_checkpointer)
        logger.debug(
            "checkpointer_acquired",
            type=type(checkpointer).__name__,
        )

    # 编译图
    graph = builder.compile(checkpointer=checkpointer)

    logger.info(
        "chat_graph_compiled",
        checkpointer_type=type(checkpointer).__name__,
        max_iterations=max_iterations,
    )

    return graph


def _has_tool_calls(state: ChatState) -> bool:
    """检查状态中是否有工具调用

    Args:
        state: 当前状态

    Returns:
        是否有工具调用
    """
    messages = state.get("messages", [])
    if not messages:
        return False

    last_message = messages[-1]
    return hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)


async def invoke_chat_graph(
    message: str,
    session_id: str,
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    tenant_id: int | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    use_postgres_checkpointer: bool = True,
) -> list[BaseMessage]:
    """调用聊天图（便捷函数）

    Args:
        message: 用户消息
        session_id: 会话 ID（作为 thread_id）
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器（如果提供，则不使用默认 checkpointer）
        tenant_id: 租户 ID
        max_iterations: 最大迭代次数
        use_postgres_checkpointer: 是否使用 PostgreSQL checkpointer

    Returns:
        消息列表

    Examples:
        ```python
        messages = await invoke_chat_graph(
            message="你好",
            session_id="session-123",
        )
        ```
    """
    graph = await compile_chat_graph(
        llm_service=llm_service,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        tenant_id=tenant_id,
        max_iterations=max_iterations,
        use_postgres_checkpointer=use_postgres_checkpointer,
    )

    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config,
    )

    return result.get("messages", [])


async def stream_chat_graph(
    message: str,
    session_id: str,
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    tenant_id: int | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    use_postgres_checkpointer: bool = True,
):
    """流式调用聊天图（便捷函数）

    Args:
        message: 用户消息
        session_id: 会话 ID（作为 thread_id）
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器（如果提供，则不使用默认 checkpointer）
        tenant_id: 租户 ID
        max_iterations: 最大迭代次数
        use_postgres_checkpointer: 是否使用 PostgreSQL checkpointer

    Yields:
        流式事件

    Examples:
        ```python
        async for event in stream_chat_graph(
            message="你好",
            session_id="session-123",
        ):
            print(event)
        ```
    """
    graph = await compile_chat_graph(
        llm_service=llm_service,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        tenant_id=tenant_id,
        max_iterations=max_iterations,
        use_postgres_checkpointer=use_postgres_checkpointer,
    )

    config = {"configurable": {"thread_id": session_id}}

    async for event in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config,
        stream_mode="messages",
    ):
        yield event


__all__ = [
    # 构建函数
    "build_chat_graph",
    "compile_chat_graph",
    "invoke_chat_graph",
    "stream_chat_graph",
    # 常量
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MAX_ITERATIONS",
]
