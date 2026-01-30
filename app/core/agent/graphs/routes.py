"""路由函数

定义图工作流中的路由决策逻辑。
"""

from typing import Literal

from app.core.agent.state import AgentState


def route_by_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """根据是否有工具调用决定路由

    检查最后一条消息是否有工具调用，决定下一步是执行工具还是结束。

    Args:
        state: 当前状态

    Returns:
        下一个节点名称："tools" 或 "__end__"
    """
    last_message = state["messages"][-1]
    # 检查最后一条消息是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"
