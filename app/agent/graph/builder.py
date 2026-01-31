"""图构建函数

提供 LangGraph 工作流的构建函数。

遵循 DeerFlow 风格：使用 StateGraph 直接构建，不使用抽象类。
"""

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.agent.graph.nodes import chat_node, route_by_tools, tools_node
from app.agent.graph.types import ChatState
from app.config.settings import get_settings
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 图构建函数 ==============


def build_chat_graph(
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
) -> StateGraph:
    """构建聊天图（未编译）

    遵循 DeerFlow 风格：直接使用 StateGraph，不使用抽象类。

    图结构：
        START -> chat -> route_by_tools -> tools or END
        tools -> chat

    Args:
        llm_service: LLM 服务实例（用于默认系统提示词）
        system_prompt: 系统提示词（用于默认系统提示词）

    Returns:
        StateGraph 实例（未编译）

    Examples:
        ```python
        from app.agent.graph.builder import build_chat_graph
        from langgraph.checkpoint.memory import MemorySaver

        builder = build_chat_graph()
        graph = builder.compile(checkpointer=MemorySaver())
        ```
    """
    builder = StateGraph(ChatState)

    # 添加节点
    builder.add_node("chat", chat_node)
    builder.add_node("tools", tools_node)

    # 设置入口点
    builder.add_edge(START, "chat")

    # 添加条件边：chat 节点根据是否有工具调用决定路由
    builder.add_conditional_edges(
        "chat",
        route_by_tools,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    # 工具执行后返回聊天节点
    builder.add_edge("tools", "chat")

    logger.debug("chat_graph_structure_built")
    return builder


def compile_chat_graph(
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """编译聊天图

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器

    Returns:
        编译后的 CompiledStateGraph

    Examples:
        ```python
        from app.agent.graph.builder import compile_chat_graph

        graph = compile_chat_graph()
        result = await graph.ainvoke(
            {"messages": [("user", "你好")]},
            {"configurable": {"thread_id": "session-123"}}
        )
        ```
    """
    llm_service = llm_service or get_llm_service()

    if system_prompt is None:
        system_prompt = """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

    # 构建图
    builder = build_chat_graph(llm_service, system_prompt)

    # 默认使用 MemorySaver
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.debug("using_memory_checkpointer")

    # 编译图
    graph = builder.compile(checkpointer=checkpointer)

    logger.info(
        "chat_graph_compiled",
        has_checkpointer=checkpointer is not None,
    )

    return graph


# ============== 便捷调用函数 ==============


async def invoke_chat_graph(
    message: str,
    session_id: str,
    user_id: str | None = None,
    tenant_id: int | None = None,
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> ChatState:
    """便捷函数：调用聊天图

    Args:
        message: 用户消息
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器

    Returns:
        最终状态

    Examples:
        ```python
        from app.agent.graph.builder import invoke_chat_graph

        result = await invoke_chat_graph(
            message="你好",
            session_id="session-123"
        )
        print(result["messages"][-1].content)
        ```
    """
    from app.agent.graph.types import create_state_from_input

    llm_service = llm_service or get_llm_service()

    # 设置默认系统提示词
    if system_prompt is None:
        system_prompt = """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

    # 编译图
    graph = compile_chat_graph(llm_service, system_prompt, checkpointer)

    # 准备输入状态
    input_state = create_state_from_input(
        input_data=message,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )

    # 准备配置
    config = {
        "configurable": {"thread_id": session_id},
        "metadata": {
            "llm_service": llm_service,
            "system_prompt": system_prompt,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
    }

    # 调用图
    result = await graph.ainvoke(input_state, config)

    return result


# ============== 流式调用函数 ==============


async def stream_chat_graph(
    message: str,
    session_id: str,
    user_id: str | None = None,
    tenant_id: int | None = None,
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    stream_mode: Literal["messages", "updates", "values"] = "messages",
):
    """流式调用聊天图

    Args:
        message: 用户消息
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        llm_service: LLM 服务实例
        system_prompt: 系统提示词
        checkpointer: 检查点保存器
        stream_mode: 流模式（messages, updates, values）

    Yields:
        流式数据

    Examples:
        ```python
        from app.agent.graph.builder import stream_chat_graph

        async for chunk in stream_chat_graph(
            message="你好",
            session_id="session-123",
            stream_mode="messages"
        ):
            print(chunk)
        ```
    """
    from app.agent.graph.types import create_state_from_input

    llm_service = llm_service or get_llm_service()

    if system_prompt is None:
        system_prompt = """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。"""

    graph = compile_chat_graph(llm_service, system_prompt, checkpointer)

    input_state = create_state_from_input(
        input_data=message,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )

    config = {
        "configurable": {"thread_id": session_id},
        "metadata": {
            "llm_service": llm_service,
            "system_prompt": system_prompt,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
    }

    async for chunk in graph.astream(input_state, config, stream_mode=stream_mode):
        yield chunk


__all__ = [
    # 构建函数
    "build_chat_graph",
    "compile_chat_graph",
    # 调用函数
    "invoke_chat_graph",
    "stream_chat_graph",
]
