"""图工作流模块

提供可扩展的图工作流系统，基于 LangGraph StateGraph。

使用示例:
    ```python
    from app.agent.graphs import ChatGraph, create_chat_graph, create_react_agent, create_interrupt_graph

    # 方式 1: 直接使用类
    graph = ChatGraph(llm_service=llm_service)
    graph.compile()
    response = await graph.ainvoke(input_data, config)

    # 方式 2: 使用工厂函数
    graph = create_chat_graph(system_prompt="你是一个助手")
    response = await graph.ainvoke(input_data, config)

    # 方式 3: 快速创建 ReAct Agent
    agent = create_react_agent(tools=[my_tool])
    response = await agent.get_response("你好", session_id="123")

    # 方式 4: 创建 Human-in-the-Loop 图
    graph = create_interrupt_graph(auto_interrupt=True)
    response = await graph.ainvoke(input_data, config)
    # 如果触发中断，进行审核后继续
    if response.get("_interrupted"):
        approval = {"approved": True, "feedback": "同意"}
        response = await graph.aresume(approval, config)
    ```
"""

from app.agent.graphs.base import BaseGraph
from app.agent.graphs.cache import (
    GraphCache,
    clear_graph_cache,
    get_cached_graph,
    get_graph_cache,
    get_graph_cache_stats,
)
from app.agent.graphs.chat import ChatGraph, create_chat_graph
from app.agent.graphs.interrupt import (
    HumanApproval,
    InterruptGraph,
    InterruptRequest,
    create_interrupt_graph,
)
from app.agent.graphs.nodes import chat_node, tools_node
from app.agent.graphs.react import ReactAgent, create_react_agent
from app.agent.graphs.routes import route_by_tools

__all__ = [
    # 抽象基类
    "BaseGraph",
    # 具体实现
    "ChatGraph",
    "create_chat_graph",
    # 图缓存（性能优化）
    "GraphCache",
    "get_cached_graph",
    "get_graph_cache",
    "clear_graph_cache",
    "get_graph_cache_stats",
    # ReAct Agent（快速开发选项）
    "ReactAgent",
    "create_react_agent",
    # Human-in-the-Loop
    "InterruptGraph",
    "create_interrupt_graph",
    "HumanApproval",
    "InterruptRequest",
    # 节点和路由（供扩展使用）
    "chat_node",
    "tools_node",
    "route_by_tools",
]
