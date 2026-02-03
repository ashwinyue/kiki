"""图节点函数

定义 LangGraph 工作流中使用的节点函数。

使用 LangGraph 标准模式：
- chat_node 接受预配置的 LLM 和 system_prompt
- 集成上下文窗口管理，防止超过 token 限制
- 使用标准 LangGraph ToolNode 执行工具
"""

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import RunnableConfig

from app.agent.state import ChatState  # 从统一状态模块导入
from app.observability.logging import get_logger

logger = get_logger(__name__)


async def chat_node(
    state: ChatState,
    config: RunnableConfig,
    llm: Any,
    system_prompt: str,
) -> dict:
    """聊天节点 - 生成 LLM 响应

    使用预配置的 LLM（已绑定工具），无需再次获取工具。
    自动应用上下文窗口管理，防止超过 token 限制。

    Args:
        state: 当前状态
        config: 运行配置
        llm: 预配置的 LLM 实例（已绑定工具）
        system_prompt: 系统提示词

    Returns:
        状态更新字典
    """
    logger.debug(
        "chat_node_entered",
        message_count=len(state.get("messages", [])),
        iteration_count=state.get("iteration_count", 0),
    )

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    if iteration_count >= max_iterations:
        logger.warning("max_iterations_reached", count=iteration_count)
        return {
            "messages": [
                AIMessage(
                    content="抱歉，已达到最大迭代次数。请重新开始对话。"
                )
            ],
            "error": "max_iterations_reached",
        }

    messages = state["messages"]

    try:
        from app.agent.state import ChatState as ChatStateClass
        from app.agent.state import ChatState as ChatStateClass

        state_obj = ChatStateClass(**state)
        messages = state_obj.trim_messages()

        if len(messages) < len(state["messages"]):
            logger.info(
                "messages_trimmed",
                original_count=len(state["messages"]),
                trimmed_count=len(messages),
            )
    except Exception as e:
        logger.warning("message_trimming_failed", error=str(e))
        # 如果截断失败，继续使用原始消息列表

    if not messages or not isinstance(messages[0], SystemMessage):
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
    else:
        messages_with_system = messages

    # 构建提示词模板（不包含 system，因为已经在消息中）
    prompt_template = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages")]
    )

    chain = prompt_template | llm

    try:
        # 调用 LLM
        response = await chain.ainvoke(
            {"messages": messages_with_system},
            config,
        )

        logger.info(
            "llm_response_generated",
            has_tool_calls=bool(
                hasattr(response, "tool_calls") and response.tool_calls
            ),
            iteration_count=iteration_count + 1,
        )

        # 返回状态更新
        return {
            "messages": [response],
            "iteration_count": iteration_count + 1,
        }

    except Exception as e:
        logger.exception("llm_call_failed", error=str(e))
        error_message = AIMessage(content=f"抱歉，处理您的请求时出错：{str(e)}")
        return {
            "messages": [error_message],
            "error": str(e),
        }


__all__ = ["chat_node"]
