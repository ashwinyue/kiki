"""节点函数

定义图工作流中的可复用节点函数。
"""

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import RunnableConfig

from app.agent.state import AgentState
from app.observability.logging import get_logger


def _get_tool_node():
    """延迟导入 ToolNode 避免循环导入"""
    from app.agent.tools import get_tool_node as _get_tool_node_impl

    return _get_tool_node_impl()


if TYPE_CHECKING:
    from app.llm import LLMService


logger = get_logger(__name__)


def create_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """构建提示词模板

    使用 ChatPromptTemplate 替代手动字符串拼接，
    支持变量替换和消息占位符。

    Args:
        system_prompt: 系统提示词

    Returns:
        ChatPromptTemplate 实例
    """
    messages = [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
    return ChatPromptTemplate.from_messages(messages)


def get_llm_with_tools(
    llm_service: "LLMService",
    tools: list | None = None,
) -> BaseChatModel | None:
    """获取绑定工具后的 LLM

    使用 LLMService.get_llm_with_tools() 方法，
    该方法确保在正确的顺序下绑定工具并应用重试。

    Args:
        llm_service: LLM 服务实例
        tools: 工具列表（可选，未提供时获取所有可用工具）

    Returns:
        绑定工具的 LLM 实例
    """
    if tools is None:
        from app.agent.tools import list_tools

        tools = list_tools()

    return llm_service.get_llm_with_tools(tools)


def create_chat_node_factory(
    llm_service: "LLMService",
    system_prompt: str,
):
    """创建聊天节点工厂函数

    返回一个配置好的聊天节点函数，闭包捕获 LLM 服务和提示词。

    Args:
        llm_service: LLM 服务实例
        system_prompt: 系统提示词

    Returns:
        聊天节点函数
    """

    async def chat_node(
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """聊天节点 - 生成 LLM 响应

        使用 ChatPromptTemplate 构建 LLM 输入，
        使用 conditional_edges 进行路由决策。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新
        """
        logger.debug("chat_node_entered", message_count=len(state["messages"]))

        try:
            llm = get_llm_with_tools(llm_service, [])
            if not llm:
                raise RuntimeError("LLM 未初始化")

            # 构建提示词模板
            prompt_template = create_prompt_template(system_prompt)

            # 构建 LCEL 链：prompt | llm
            chain = prompt_template | llm

            # 调用链
            response = await chain.ainvoke(
                {
                    "system_prompt": system_prompt,
                    "messages": state["messages"],
                },
                config,
            )

            logger.info(
                "llm_response_generated",
                model=llm_service.current_model,
                has_tool_calls=bool(hasattr(response, "tool_calls") and response.tool_calls),
            )

            return {"messages": [response]}

        except Exception as e:
            logger.exception("llm_call_failed", error=str(e))
            # 返回错误消息
            error_message = AIMessage(content=f"抱歉，处理您的请求时出错：{str(e)}")
            return {"messages": [error_message]}

    return chat_node


async def tools_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """工具节点 - 执行工具调用

    使用 LangGraph 的 ToolNode 执行工具调用。

    Args:
        state: 当前状态
        config: 运行配置

    Returns:
        状态更新
    """
    logger.debug("tools_node_entered")

    # 使用 ToolNode 执行工具调用
    tool_node = _get_tool_node()
    result = await tool_node.ainvoke(state, config)

    logger.info(
        "tools_executed",
        tool_result_count=len(result.get("messages", [])),
    )

    return result


# 导出便捷函数
def chat_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """默认聊天节点（需在使用时配置 LLM）

    注意：此函数需要配合 create_chat_node_factory 使用，
    或在具体图实现中自定义。
    """
    raise NotImplementedError(
        "请使用 create_chat_node_factory(llm_service, system_prompt) 创建配置好的聊天节点"
    )
