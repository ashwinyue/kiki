"""图工作流模块

提供可扩展的图工作流系统，基于 LangGraph StateGraph。

使用示例:
    ```python
    from app.core.agent.graphs import ChatGraph, create_chat_graph

    # 方式 1: 直接使用类
    graph = ChatGraph(llm_service=llm_service)
    graph.compile()
    response = await graph.ainvoke(input_data, config)

    # 方式 2: 使用工厂函数
    graph = create_chat_graph(system_prompt="你是一个助手")
    response = await graph.ainvoke(input_data, config)
    ```
"""

from app.core.agent.graphs.base import BaseGraph
from app.core.agent.graphs.chat import ChatGraph, create_chat_graph
from app.core.agent.graphs.nodes import chat_node, tools_node
from app.core.agent.graphs.routes import route_by_tools

__all__ = [
    # 抽象基类
    "BaseGraph",
    # 具体实现
    "ChatGraph",
    "create_chat_graph",
    # 节点和路由（供扩展使用）
    "chat_node",
    "tools_node",
    "route_by_tools",
]
