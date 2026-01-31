"""图节点函数

定义 LangGraph 工作流中使用的节点函数。

遵循 LangGraph 最佳实践：
- 节点函数返回 Command 进行跳转
- 支持 ChatState 和 AgentState
"""

from typing import TYPE_CHECKING, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RunnableConfig

from app.agent.graph.types import ChatState
from app.agent.graph.utils import get_message_content
from app.agent.tools import alist_tools
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.llm import LLMService

logger = get_logger(__name__)


# ============== 聊天节点 ==============


async def chat_node(
    state: ChatState,
    config: RunnableConfig,
) -> Command:
    """聊天节点 - 生成 LLM 响应

    使用 Command 模式进行节点跳转和状态更新。

    Args:
        state: 当前状态
        config: 运行配置，metadata 中需包含 llm_service 和 system_prompt

    Returns:
        Command 指令，包含状态更新和跳转目标
    """
    logger.debug("chat_node_entered", message_count=len(state.get("messages", [])))

    # 从 config 中获取 llm_service 和 system_prompt
    metadata = config.get("metadata", {})
    llm_service = metadata.get("llm_service")
    system_prompt = metadata.get("system_prompt")

    if not llm_service:
        raise RuntimeError("LLM service not found in config metadata")

    # 获取带工具绑定的 LLM 实例
    llm = llm_service.get_llm_with_tools([])
    if not llm:
        raise RuntimeError("Failed to get LLM from LLMService")

    # 构建提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 构建 LCEL 链
    chain = prompt_template | llm

    try:
        # 调用 LLM
        response = await chain.ainvoke({
            "system_prompt": system_prompt,
            "messages": state["messages"],
        }, config)

        logger.info(
            "llm_response_generated",
            has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
        )

        # 递增迭代计数
        iteration_count = state.get("iteration_count", 0) + 1

        # 检查是否有工具调用，决定下一步
        if hasattr(response, "tool_calls") and response.tool_calls:
            return Command(
                update={
                    "messages": [response],
                    "iteration_count": iteration_count,
                },
                goto="tools",
            )
        else:
            return Command(
                update={
                    "messages": [response],
                    "iteration_count": iteration_count,
                },
                goto="__end__",
            )

    except Exception as e:
        logger.exception("llm_call_failed", error=str(e))
        error_message = AIMessage(content=f"抱歉，处理您的请求时出错：{str(e)}")
        return Command(
            update={
                "messages": [error_message],
                "error": str(e),
            },
            goto="__end__",
        )


# ============== 工具节点 ==============


async def tools_node(
    state: ChatState,
    config: RunnableConfig,
) -> Command:
    """工具节点 - 执行工具调用

    使用 LangGraph 的 ToolNode 执行工具调用。
    执行完成后返回 chat 节点。

    Args:
        state: 当前状态
        config: 运行配置

    Returns:
        Command 指令，执行完成后返回 chat 节点
    """
    logger.debug("tools_node_entered")

    # 获取租户 ID
    tenant_id = None
    if isinstance(config, dict):
        tenant_id = config.get("metadata", {}).get("tenant_id")
    else:
        metadata = getattr(config, "metadata", None) or {}
        tenant_id = metadata.get("tenant_id")

    # 获取工具列表
    tools = await alist_tools(include_mcp=True, tenant_id=tenant_id)

    # 创建 ToolNode 并执行
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke(state, config)

    logger.info(
        "tools_executed",
        tool_result_count=len(result.get("messages", [])),
    )

    # 返回 chat 节点继续循环
    return Command(
        update=result,
        goto="chat",
    )


# ============== 路由函数 ==============


def route_by_tools(state: ChatState) -> Literal["tools", "__end__"]:
    """路由函数：根据是否有工具调用决定下一步

    Args:
        state: 当前状态

    Returns:
        下一个节点名称
    """
    # 检查迭代次数，防止无限循环
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    if iteration_count >= max_iterations:
        logger.warning(
            "max_iterations_reached",
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        return "__end__"

    # 检查最后一条消息是否有工具调用
    if not state["messages"]:
        return "__end__"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


# ============== 节点工厂 ==============


def create_chat_node_factory(
    llm_service: "LLMService",
    system_prompt: str,
) -> callable:
    """创建聊天节点工厂函数

    返回一个配置好的聊天节点函数，闭包捕获 LLM 服务和提示词。

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词

    Returns:
        聊天节点函数

    Examples:
        ```python
        from app.llm import get_llm_service
        from app.agent.graph.nodes import create_chat_node_factory

        llm_service = get_llm_service()
        chat_node = create_chat_node_factory(
            llm_service,
            "你是一个有用的助手。"
        )

        # 在图中使用
        builder.add_node("chat", chat_node)
        ```
    """
    llm_with_tools = llm_service.get_llm_with_tools([])

    async def chat_node_impl(
        state: ChatState,
        config: RunnableConfig,
    ) -> Command:
        """聊天节点实现"""
        logger.debug("chat_node_entered", message_count=len(state.get("messages", [])))

        try:
            # 构建提示词模板
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="messages"),
            ])

            # 构建 LCEL 链
            chain = prompt_template | llm_with_tools

            # 调用 LLM
            response = await chain.ainvoke(
                {
                    "system_prompt": system_prompt,
                    "messages": state["messages"],
                },
                config,
            )

            logger.info(
                "llm_response_generated",
                has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
            )

            # 根据是否有工具调用决定路由
            if hasattr(response, "tool_calls") and response.tool_calls:
                return Command(
                    update={"messages": [response]},
                    goto="tools",
                )
            else:
                return Command(
                    update={"messages": [response]},
                    goto="__end__",
                )

        except Exception as e:
            logger.exception("llm_call_failed", error=str(e))
            error_message = AIMessage(content=f"抱歉，处理您的请求时出错：{str(e)}")
            return Command(
                update={"messages": [error_message]},
                goto="__end__",
            )

    return chat_node_impl


__all__ = [
    # 节点函数
    "chat_node",
    "tools_node",
    # 路由函数
    "route_by_tools",
    # 工厂函数
    "create_chat_node_factory",
]
