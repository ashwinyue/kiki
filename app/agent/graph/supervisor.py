"""Supervisor-Agent 多 Agent 编排模式

实现 Supervisor-Agent 协作模式，支持复杂任务分解和多 Agent 协作。

参考外部项目的 supervisor_agent 设计模式。

架构说明：
    Supervisor 节点负责任务分解和路由
    各个专门 Agent 负责具体任务执行
    支持动态任务路由和迭代控制

图结构：
    START -> Supervisor -> [Researcher, Scrapper, Database]
    [Researcher, Scrapper, Database] -> Supervisor
    Supervisor -> FINISH -> END
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.graph.types import ChatState, increment_iteration
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 状态定义 ==============


class RouteResponse(BaseModel):
    """Supervisor 路由决策

    使用 Pydantic 模型确保 LLM 输出可解析。
    """

    next: Literal["Researcher", "Scrapper", "Database", "FINISH"] = Field(
        description="下一个要调用的 Agent"
    )
    reasoning: str = Field(description="路由决策的推理过程")


class SupervisorState(ChatState):
    """Supervisor 状态（扩展 ChatState）

    Attributes:
        messages: 消息列表
        next: 下一个要调用的 Agent
        reasoning: 路由决策的推理
        task_completed: 任务是否完成
        agent_results: 各 Agent 的执行结果
    """

    # 路由决策
    next: Literal["Researcher", "Scrapper", "Database", "FINISH"] | None = None
    reasoning: str | None = None
    task_completed: bool = False

    # Agent 执行结果
    agent_results: dict[str, Any] = {}

    # 当前迭代信息
    current_agent: str | None = None
    agent_history: list[str] = []  # 记录已调用的 Agent


# ============== Supervisor 节点 ==============


async def supervisor_node(state: SupervisorState) -> dict[str, Any]:
    """Supervisor 节点 - 决定调用哪个 Agent

    分析对话历史，决定路由到哪个专门 Agent 或完成任务。

    Args:
        state: 当前状态

    Returns:
        状态更新（包含 next 和 reasoning）
    """
    llm_service = get_llm_service()
    llm = llm_service.get_llm()

    # 获取最新的用户消息
    last_message = state["messages"][-1] if state["messages"] else None

    if not last_message:
        return {
            "next": "FINISH",
            "reasoning": "没有消息，结束对话",
            "task_completed": True,
        }

    # 构建 Supervisor 提示词
    supervisor_prompt = """你是任务协调者（Supervisor），负责将用户任务分配给合适的专家 Agent。

可用的专家 Agent：
1. **Researcher** - 网络搜索、学术查询、信息收集
2. **Scrapper** - 网页抓取、数据提取、内容解析
3. **Database** - 数据库查询、数据分析、报告生成

分析用户需求，选择最合适的 Agent 来处理任务。如果任务已完成或需要结束对话，选择 FINISH。

决策指南：
- 需要搜索网络或学术资源 → Researcher
- 需要从网页提取数据 → Scrapper
- 需要查询数据库或分析数据 → Database
- 任务已完成或无法处理 → FINISH

当前对话历史：
"""

    # 添加最近的消息到提示词
    recent_messages = state["messages"][-6:]  # 最近 6 条消息
    for msg in recent_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        supervisor_prompt += f"\n{role}: {msg.content[:200]}..."  # 限制长度

    # 添加 Agent 执行历史
    if state["agent_history"]:
        supervisor_prompt += f"\n\n已调用的 Agent: {', '.join(state['agent_history'])}"

    # 构建结构化输出链
    supervisor_chain = (
        SystemMessage(content=supervisor_prompt)
        | AIMessage(content="")
        | llm.with_structured_output(RouteResponse)
    )

    try:
        result: RouteResponse = await supervisor_chain.ainvoke(state["messages"])
        logger.info(
            "supervisor_route_decision",
            next=result.next,
            reasoning=result.reasoning[:100],  # 限制日志长度
        )
        return {
            "next": result.next,
            "reasoning": result.reasoning,
            "task_completed": result.next == "FINISH",
        }
    except Exception as e:
        logger.error("supervisor_decision_failed", error=str(e))
        # 失败时默认结束
        return {
            "next": "FINISH",
            "reasoning": "决策失败，结束对话",
            "task_completed": True,
        }


# ============== 专门 Agent 节点 ==============


async def researcher_agent(state: SupervisorState) -> dict[str, Any]:
    """Researcher Agent - 网络搜索和学术查询

    Args:
        state: 当前状态

    Returns:
        状态更新
    """
    logger.info("researcher_agent_invoked")

    # 这里可以调用搜索工具
    # 目前返回模拟结果
    result = {
        "agent": "Researcher",
        "content": "Researcher Agent 执行完成",
        "data": {},
    }

    # 更新状态
    return {
        "current_agent": "Researcher",
        "agent_results": {**state.get("agent_results", {}), "researcher": result},
        "agent_history": state.get("agent_history", []) + ["Researcher"],
    }


async def scrapper_agent(state: SupervisorState) -> dict[str, Any]:
    """Scrapper Agent - 网页抓取和数据提取

    Args:
        state: 当前状态

    Returns:
        状态更新
    """
    logger.info("scrapper_agent_invoked")

    # 这里可以调用抓取工具
    result = {
        "agent": "Scrapper",
        "content": "Scrapper Agent 执行完成",
        "data": {},
    }

    return {
        "current_agent": "Scrapper",
        "agent_results": {**state.get("agent_results", {}), "scrapper": result},
        "agent_history": state.get("agent_history", []) + ["Scrapper"],
    }


async def database_agent(state: SupervisorState) -> dict[str, Any]:
    """Database Agent - 数据库查询和分析

    Args:
        state: 当前状态

    Returns:
        状态更新
    """
    logger.info("database_agent_invoked")

    # 这里可以调用数据库工具
    result = {
        "agent": "Database",
        "content": "Database Agent 执行完成",
        "data": {},
    }

    return {
        "current_agent": "Database",
        "agent_results": {**state.get("agent_results", {}), "database": result},
        "agent_history": state.get("agent_history", []) + ["Database"],
    }


# ============== 路由函数 ==============


def route_from_supervisor(state: SupervisorState) -> str:
    """根据 Supervisor 决策路由到对应 Agent

    Args:
        state: 当前状态

    Returns:
        目标节点名称
    """
    next_agent = state.get("next")
    if next_agent == "FINISH":
        return END
    return next_agent


def should_continue_supervisor(state: SupervisorState) -> str:
    """检查是否应该继续 Supervisor 循环

    Args:
        state: 当前状态

    Returns:
        "continue" 或 "end"
    """
    # 检查迭代次数
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    if iteration_count >= max_iterations:
        logger.info("supervisor_max_iterations_reached")
        return END

    # 检查任务是否完成
    if state.get("task_completed", False):
        logger.info("supervisor_task_completed")
        return END

    return "continue"


# ============== 图构建函数 ==============


def build_supervisor_graph(
    llm_service: LLMService | None = None,
) -> StateGraph:
    """构建 Supervisor-Agent 协作图（未编译）

    图结构：
        START -> Supervisor -> [Researcher, Scrapper, Database]
        [Researcher, Scrapper, Database] -> Supervisor
        Supervisor -> FINISH -> END

    Args:
        llm_service: LLM 服务实例

    Returns:
        StateGraph 实例（未编译）

    Examples:
        ```python
        from app.agent.graph.supervisor import build_supervisor_graph
        from langgraph.checkpoint.memory import MemorySaver

        builder = build_supervisor_graph()
        graph = builder.compile(checkpointer=MemorySaver())

        result = await graph.ainvoke(
            {"messages": [("user", "帮我搜索最新 AI 技术趋势")]},
            {"configurable": {"thread_id": "session-123"}}
        )
        ```
    """
    builder = StateGraph(SupervisorState)

    # 添加节点
    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Researcher", researcher_agent)
    builder.add_node("Scrapper", scrapper_agent)
    builder.add_node("Database", database_agent)

    # 设置入口点
    builder.add_edge(START, "Supervisor")

    # Supervisor -> 专门 Agent
    builder.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "Researcher": "Researcher",
            "Scrapper": "Scrapper",
            "Database": "Database",
            END: END,
        },
    )

    # 专门 Agent -> Supervisor（循环）
    builder.add_edge("Researcher", "Supervisor")
    builder.add_edge("Scrapper", "Supervisor")
    builder.add_edge("Database", "Supervisor")

    logger.debug("supervisor_graph_structure_built")
    return builder


async def compile_supervisor_graph(
    llm_service: LLMService | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """编译 Supervisor-Agent 协作图

    Args:
        llm_service: LLM 服务实例
        checkpointer: 检查点保存器

    Returns:
        编译后的 CompiledStateGraph
    """
    from langgraph.checkpoint.memory import MemorySaver

    llm_service = llm_service or get_llm_service()

    # 构建图
    builder = build_supervisor_graph(llm_service)

    # 默认使用 MemorySaver
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.debug("using_memory_checkpointer")

    # 编译图
    graph = builder.compile(checkpointer=checkpointer)

    logger.info(
        "supervisor_graph_compiled",
        has_checkpointer=checkpointer is not None,
    )

    return graph


# ============== 便捷调用函数 ==============


async def invoke_supervisor(
    message: str,
    session_id: str,
    user_id: str | None = None,
    tenant_id: int | None = None,
    llm_service: LLMService | None = None,
) -> SupervisorState:
    """便捷函数：调用 Supervisor-Agent 图

    Args:
        message: 用户消息
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        llm_service: LLM 服务实例

    Returns:
        最终状态

    Examples:
        ```python
        from app.agent.graph.supervisor import invoke_supervisor

        result = await invoke_supervisor(
            message="帮我搜索最新的 AI 技术趋势",
            session_id="session-123"
        )
        print(result["agent_results"])
        ```
    """
    from app.agent.graph.types import create_state_from_input
    from langchain_core.messages import HumanMessage

    # 编译图
    graph = await compile_supervisor_graph(llm_service)

    # 准备输入状态
    input_state = SupervisorState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        session_id=session_id,
        tenant_id=tenant_id,
        iteration_count=0,
        max_iterations=10,
        next=None,
        reasoning=None,
        task_completed=False,
        agent_results={},
        current_agent=None,
        agent_history=[],
    )

    # 准备配置
    config = {
        "configurable": {"thread_id": session_id},
        "metadata": {
            "llm_service": llm_service,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
    }

    # 调用图
    result = await graph.ainvoke(input_state, config)

    return result


__all__ = [
    # 状态和模型
    "RouteResponse",
    "SupervisorState",
    # 节点函数
    "supervisor_node",
    "researcher_agent",
    "scrapper_agent",
    "database_agent",
    # 路由函数
    "route_from_supervisor",
    "should_continue_supervisor",
    # 构建函数
    "build_supervisor_graph",
    "compile_supervisor_graph",
    "invoke_supervisor",
]
