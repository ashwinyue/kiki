"""Multi-Agent 状态定义

统一的多 Agent 状态，整合了 SupervisorState 和 MultiAgentState。
所有多 Agent 模式（Supervisor、Router、Hierarchical）都使用此状态。

设计原则:
    - 单一职责: 只定义状态数据结构
    - 开闭原则: 易于扩展新的多 Agent 模式
    - 依赖倒置: 依赖抽象的 ChatState

使用示例:
    ```python
    from app.agent.state.multi_agent import MultiAgentState
    from langchain_core.messages import HumanMessage

    state = MultiAgentState(
        messages=[HumanMessage(content="搜索最新新闻")],
        next_agent="search-agent",
        agent_outputs={},
    )
    ```
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.agent.state.chat import ChatState


class MultiAgentState(ChatState):
    """统一的多 Agent 状态

    整合了 SupervisorState 和 MultiAgentState 的所有字段。

    Attributes:
        # ========== 路由相关 ==========
        next_agent: 下一个要调用的 Agent ID
        routing_reasoning: 路由决策的推理过程

        # ========== Agent 输出 ==========
        agent_outputs: 各 Agent 的执行结果 {agent_id: output}

        # ========== 调用链追踪 ==========
        current_agent_role: 当前 Agent 角色（supervisor, worker, etc.）
        parent_execution_id: 父执行记录 ID（用于调用链追踪）
        current_execution_id: 当前执行记录 ID

        # ========== 迭代控制 ==========
        task_completed: 任务是否完成
        agent_history: 已调用的 Agent 列表

        # ========== 当前迭代信息 ==========
        current_agent: 当前正在执行的 Agent ID（与 next_agent 配合使用）

    Examples:
        ```python
        # 初始化状态
        state = MultiAgentState(
            messages=[HumanMessage(content="帮我搜索 AI 新闻")],
            next_agent="search-agent",
            agent_outputs={},
            iteration_count=0,
            max_iterations=10,
        )

        # 更新状态（Supervisor 节点）
        state["next_agent"] = "rag-agent"
        state["routing_reasoning"] = "需要查询知识库"

        # 更新状态（Worker 节点）
        state["agent_outputs"]["search-agent"] = {"content": "搜索结果..."}
        state["agent_history"].append("search-agent")
        ```
    """

    # ========== 路由相关 ==========
    next_agent: str | None
    routing_reasoning: str | None

    # ========== Agent 输出 ==========
    agent_outputs: dict[str, Any]

    # ========== 调用链追踪 ==========
    current_agent_role: str | None  # supervisor, worker, router, etc.
    parent_execution_id: UUID | None
    current_execution_id: UUID | None

    # ========== 迭代控制 ==========
    task_completed: bool
    agent_history: list[str]

    # ========== 当前迭代信息 ==========
    current_agent: str | None


__all__ = ["MultiAgentState"]
