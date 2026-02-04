"""Agent Workflow - 参考 DeerFlow 设计

提供完整的 Agent 工作流执行功能。

架构：
- graph/builder.py: 构建工作流图
- workflow.py: 执行工作流

使用示例：
    ```python
    from app.agent.workflow import run_agent_workflow

    result = await run_agent_workflow(
        user_input="创建一个快速排序算法",
        max_step_num=3,
    )
    ```
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.graph.agent_factory import create_agent
from app.agent.graph.agents import AGENT_REGISTRY
from app.observability.logging import get_logger

logger = get_logger(__name__)


async def coordinator_node(state: dict, config: dict) -> dict:
    """协调者节点 - 处理入口任务"""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info("coordinator_processing", user_input=user_input[:100])
    return {"research_topic": user_input, "clarified_research_topic": user_input}


async def planner_node(state: dict, config: dict) -> dict:
    """规划者节点 - 任务分解和计划生成"""
    research_topic = state.get("research_topic", "")
    background_results = state.get("background_investigation_results")

    planner_config = AGENT_REGISTRY["planner"]
    planner = create_agent(
        agent_name="planner",
        agent_type=planner_config["agent_type"],
        tools=planner_config["tools"](),
        prompt_template=planner_config["prompt_template"],
    )

    input_messages = state.get("messages", [])
    if research_topic:
        input_messages = [
            {"role": "user", "content": f"目标：{research_topic}"}
        ]
    if background_results:
        input_messages.append(
            {"role": "system", "content": f"背景信息：{background_results}"}
        )

    _ = await planner.ainvoke(
        {"messages": input_messages},
        {"configurable": {"thread_id": config.get("thread_id", "default")}},
    )

    return {
        "current_plan": _create_plan(research_topic),
        "plan_iterations": state.get("plan_iterations", 0) + 1,
    }


def _create_plan(research_topic: str) -> dict:
    """创建默认计划结构

    TODO: 实际应该从 planner 的输出中解析结构化计划
    """
    return {
        "goal": research_topic,
        "steps": [
            {"step_type": "research", "description": "搜索相关信息", "execution_res": None},
            {"step_type": "analysis", "description": "分析数据", "execution_res": None},
            {"step_type": "processing", "description": "生成代码", "execution_res": None},
        ],
    }


async def research_team_node(state: dict, config: dict) -> dict:
    """研究团队节点 - 根据计划路由到具体 agent"""
    current_plan = state.get("current_plan")
    if not current_plan:
        return {"next_agent": "planner"}

    for step in current_plan.steps:
        if not step.execution_res:
            step_type = step.step_type

            if step_type == "research":
                return await _run_researcher(state, config)
            elif step_type == "analysis":
                return await _run_analyst(state, config)
            elif step_type == "processing":
                return await _run_coder(state, config)

    return {"next_agent": "reporter"}


async def _run_researcher(state: dict, config: dict) -> dict:
    """运行研究员 agent"""
    config = AGENT_REGISTRY["researcher"]
    agent = create_agent(
        agent_name="researcher",
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )

    research_topic = state.get("clarified_research_topic", state.get("research_topic", ""))

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"搜索：{research_topic}"}]},
        {"configurable": {"thread_id": config.get("thread_id", "default")}},
    )

    content = result["messages"][-1].content if result["messages"] else ""
    return {"research_results": content}


async def _run_analyst(state: dict, config: dict) -> dict:
    """运行分析师 agent"""
    config = AGENT_REGISTRY["analyst"]
    agent = create_agent(
        agent_name="analyst",
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )

    research_results = state.get("research_results", "")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"分析以下数据：{research_results}"}]},
        {"configurable": {"thread_id": config.get("thread_id", "default")}},
    )

    content = result["messages"][-1].content if result["messages"] else ""
    return {"analysis_results": content}


async def _run_coder(state: dict, config: dict) -> dict:
    """运行代码专家 agent"""
    config = AGENT_REGISTRY["coder"]
    agent = create_agent(
        agent_name="coder",
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )

    research_results = state.get("research_results", "")
    analysis_results = state.get("analysis_results", "")

    prompt = "生成代码\n"
    if research_results:
        prompt += f"研究信息：{research_results}\n"
    if analysis_results:
        prompt += f"分析结果：{analysis_results}\n"

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        {"configurable": {"thread_id": config.get("thread_id", "default")}},
    )

    content = result["messages"][-1].content if result["messages"] else ""
    return {"code_results": content}


async def reporter_node(state: dict, config: dict) -> dict:
    """报告员节点 - 聚合结果生成报告"""
    config = AGENT_REGISTRY["reporter"]
    agent = create_agent(
        agent_name="reporter",
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )

    outputs = {}
    if "research_results" in state:
        outputs["researcher"] = state["research_results"]
    if "analysis_results" in state:
        outputs["analyst"] = state["analysis_results"]
    if "code_results" in state:
        outputs["coder"] = state["code_results"]

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "生成报告"}], **outputs},
        {"configurable": {"thread_id": config.get("thread_id", "default")}},
    )

    return {"final_report": result["messages"][-1].content if result["messages"] else ""}


def continue_to_research_team(state: dict) -> str:
    """决定下一步的 agent"""
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    if all(step.execution_res for step in current_plan.steps):
        return "reporter"

    for step in current_plan.steps:
        if not step.execution_res:
            return "research_team"

    return "planner"


def _build_base_graph() -> StateGraph:
    """构建基础工作流图"""
    builder = StateGraph(dict)

    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "planner")
    builder.add_conditional_edges("planner", continue_to_planner)
    builder.add_conditional_edges("research_team", continue_to_research_team)
    builder.add_edge("reporter", END)

    return builder


def continue_to_planner(state: dict) -> str:
    """从 planner 的路由"""
    current_plan = state.get("current_plan")
    if current_plan and current_plan.steps:
        return "research_team"
    return "reporter"


def build_graph() -> Any:
    """构建并返回工作流图

    Returns:
        CompiledStateGraph
    """
    builder = _build_base_graph()

    return builder.compile()


graph = build_graph()


async def run_agent_workflow(
    user_input: str,
    max_step_num: int = 3,
    thread_id: str = "default",
) -> dict:
    """运行 Agent 工作流

    Args:
        user_input: 用户输入
        max_step_num: 最大执行步数
        thread_id: 线程 ID

    Returns:
        最终状态字典

    示例：
        ```python
        result = await run_agent_workflow(
            user_input="创建一个快速排序算法",
            max_step_num=3,
        )
        print(result["final_report"])
        ```
    """
    if not user_input:
        raise ValueError("用户输入不能为空")

    logger.info(
        "workflow_started",
        user_input=user_input[:100],
        max_step_num=max_step_num,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "research_topic": user_input,
        "clarified_research_topic": user_input,
    }

    final_state = None
    async for event in graph.astream(
        initial_state, config, stream_mode="values"
    ):
        final_state = event

    logger.info("workflow_completed")

    return final_state or {}


__all__ = [
    "graph",
    "run_agent_workflow",
]
