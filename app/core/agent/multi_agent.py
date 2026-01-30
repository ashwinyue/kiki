"""多 Agent 系统

支持多种多 Agent 模式：
1. Router Agent - 路由到不同的子 Agent
2. Supervisor Agent - 监督多个 Worker Agent
3. Handoff - Agent 之间动态切换

使用 LangChain 的 with_structured_output 实现类型安全的路由决策。
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command
from langgraph.types import RunnableConfig

from app.core.agent.schemas import RouteDecision, SupervisorDecision
from app.core.agent.state import AgentState
from app.core.llm import LLMService
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============== 模式 1: Router Agent ==============

class RouterAgent:
    """路由 Agent

    根据用户意图将请求路由到不同的子 Agent。

    示意图：
                    ┌─────────────┐
                    │   Router    │
                    │   Agent     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │  Sales  │  │ Support │  │ General │
        │  Agent  │  │  Agent  │  │  Agent  │
        └─────────┘  └─────────┘  └─────────┘
    """

    def __init__(
        self,
        llm_service: LLMService,
        agents: dict[str, Any],
        router_prompt: str | None = None,
    ) -> None:
        """初始化路由 Agent

        Args:
            llm_service: LLM 服务
            agents: 子 Agent 字典 {name: AgentGraph}
            router_prompt: 路由提示词
        """
        self._llm_service = llm_service
        self._agents = agents
        self._router_prompt = router_prompt or self._default_router_prompt()
        self._graph: StateGraph | None = None

        # 构建结构化输出 LLM
        self._structured_llm = self._build_structured_llm()

    def _default_router_prompt(self) -> str:
        """默认路由提示词"""
        agent_list = ", ".join(self._agents.keys())
        return f"""你是一个路由助手，负责将用户请求路由到合适的子 Agent。

可用的子 Agent:
{agent_list}

请分析用户意图，并选择最合适的 Agent 处理请求。
如果不确定，请选择 General Agent。"""

    def _build_structured_llm(self) -> BaseChatModel:
        """构建带结构化输出的 LLM

        使用 with_structured_output 确保返回类型安全的 RouteDecision。

        Returns:
            绑定结构化输出的 LLM
        """
        llm = self._llm_service.get_llm()
        if llm is None:
            raise RuntimeError("LLM 未初始化")

        return llm.with_structured_output(RouteDecision)

    def _build_router_prompt_template(self) -> ChatPromptTemplate:
        """构建路由提示词模板

        Returns:
            ChatPromptTemplate 实例
        """
        return ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_input}"),
        ])

    async def _route_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> Command:
        """路由节点 - 决定使用哪个子 Agent

        使用 with_structured_output 获取类型安全的路由决策。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            Command 对象，路由到目标 Agent
        """
        last_message = state["messages"][-1]
        user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

        try:
            # 构建提示词
            prompt_template = self._build_router_prompt_template()

            # 使用 LCEL 链调用
            chain = prompt_template | self._structured_llm
            decision: RouteDecision = await chain.ainvoke({
                "system_prompt": self._router_prompt,
                "user_input": user_input,
            })

            # 验证目标 Agent 存在
            target_agent = decision.agent
            if target_agent not in self._agents:
                logger.warning("agent_not_found", target=target_agent, fallback="list")
                target_agent = list(self._agents.keys())[0] if self._agents else "General"

            logger.info(
                "routed_to",
                target=target_agent,
                reason=decision.reason,
                confidence=decision.confidence,
            )

            # 添加路由决策消息
            decision_message = AIMessage(
                content=f"路由到 {target_agent}：{decision.reason}"
            )

            return Command(goto=target_agent, update={"messages": [decision_message]})

        except Exception as e:
            logger.exception("route_failed", error=str(e))
            # 回退到第一个可用的 Agent
            target_agent = list(self._agents.keys())[0] if self._agents else "General"
            return Command(goto=target_agent)

    def compile(self) -> StateGraph:
        """编译路由图"""
        builder = StateGraph(AgentState)

        # 添加路由节点
        builder.add_node("router", self._route_node)

        # 添加所有子 Agent 节点
        for name, agent in self._agents.items():
            # 检查 agent 是否有 _chat_node 方法
            if hasattr(agent, "_chat_node"):
                builder.add_node(name, agent._chat_node)
            else:
                # 如果是简单的 AgentGraph，包装一下
                builder.add_node(name, agent._chat_node)

        # 设置入口
        builder.set_entry_point("router")

        # 每个子 Agent 完成后结束
        for name in self._agents.keys():
            builder.add_edge(name, END)

        self._graph = builder.compile()
        return self._graph


# ============== 模式 2: Supervisor Agent ==============

class SupervisorAgent:
    """监督 Agent

    一个 Supervisor 管理多个 Worker Agent，协调它们完成任务。

    示意图：
                    ┌─────────────┐
                    │ Supervisor │◀──┐
                    │   Agent    │   │
                    └──────┬──────┘   │
                           │          │
              ┌────────────┼──────────┤
              ▼            ▼          ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │Worker 1 │  │Worker 2 │  │Worker 3 │
        └─────────┘  └─────────┘  └─────────┘
              │            │            │
              └────────────┴────────────┘
                           │
                           └───────────→ 汇报结果
    """

    def __init__(
        self,
        llm_service: LLMService,
        workers: dict[str, Any],
        supervisor_prompt: str | None = None,
    ) -> None:
        """初始化监督 Agent

        Args:
            llm_service: LLM 服务
            workers: Worker Agent 字典
            supervisor_prompt: 监督提示词
        """
        self._llm_service = llm_service
        self._workers = workers
        self._supervisor_prompt = supervisor_prompt or self._default_supervisor_prompt()
        self._graph: StateGraph | None = None

        # 构建结构化输出 LLM
        self._structured_llm = self._build_structured_llm()

    def _default_supervisor_prompt(self) -> str:
        """默认监督提示词"""
        worker_list = ", ".join(self._workers.keys())
        return f"""你是一个监督者，负责协调以下 Worker Agent 完成任务:

Worker: {worker_list}

你的职责:
1. 分析任务需求
2. 将任务分配给合适的 Worker
3. 汇总 Worker 的结果
4. 决定任务是否完成或需要更多工作"""

    def _build_structured_llm(self) -> BaseChatModel:
        """构建带结构化输出的 LLM"""
        llm = self._llm_service.get_llm()
        if llm is None:
            raise RuntimeError("LLM 未初始化")

        return llm.with_structured_output(SupervisorDecision)

    def _build_supervisor_prompt_template(self) -> ChatPromptTemplate:
        """构建监督提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("system", "当前对话历史:\n{context}"),
        ])

    async def _supervise_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> Command:
        """监督节点 - 决定下一步

        使用 with_structured_output 获取类型安全的监督决策。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            Command 对象，路由到目标 Worker 或结束
        """
        # 构建上下文
        context = "\n".join([
            f"- {m.type}: {str(m.content)[:100]}..."
            for m in state["messages"][-5:]
        ])

        try:
            # 构建提示词
            prompt_template = self._build_supervisor_prompt_template()

            # 使用 LCEL 链调用
            chain = prompt_template | self._structured_llm
            decision: SupervisorDecision = await chain.ainvoke({
                "system_prompt": self._supervisor_prompt,
                "context": context,
            })

            logger.info(
                "supervisor_decision",
                next=decision.next,
                status=decision.status,
            )

            # 根据状态决定路由
            if decision.status == "done" or decision.next == "END":
                summary_message = AIMessage(content=decision.message or "任务已完成")
                return Command(goto=END, update={"messages": [summary_message]})

            # 路由到 Worker
            if decision.next in self._workers:
                worker_message = AIMessage(content=decision.message or f"分配给 {decision.next}")
                return Command(goto=decision.next, update={"messages": [worker_message]})

            # Worker 不存在，结束
            return Command(goto=END)

        except Exception as e:
            logger.exception("supervise_failed", error=str(e))
            return Command(goto=END)

    def compile(self) -> StateGraph:
        """编译监督图"""
        builder = StateGraph(AgentState)

        # 添加监督节点
        builder.add_node("supervisor", self._supervise_node)

        # 添加 Worker 节点
        for name, agent in self._workers.items():
            if hasattr(agent, "_chat_node"):
                builder.add_node(name, agent._chat_node)
            else:
                builder.add_node(name, agent._chat_node)

        # 设置入口
        builder.set_entry_point("supervisor")

        # 每个 Worker 完成后返回监督
        for name in self._workers.keys():
            builder.add_edge(name, "supervisor")

        self._graph = builder.compile()
        return self._graph


# ============== 模式 3: Handoff Agent (Swarm) ==============

def create_handoff_tool(
    agent_name: str,
    description: str | None = None,
) -> Any:
    """创建 Agent 切换工具

    Args:
        agent_name: 目标 Agent 名称
        description: 工具描述

    Returns:
        LangChain 工具实例

    Examples:
        ```python
        transfer_to_bob = create_handoff_tool(
            agent_name="Bob",
            description="Transfer to Bob for technical support"
        )
        ```
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    actual_description = description or f"Transfer to {agent_name}"

    class HandoffInput(BaseModel):
        """Handoff 输入"""
        reason: str = Field(description="切换原因")

    async def handoff_wrapper(reason: str) -> str:
        """包装函数供 LLM 调用"""
        return f"Transferring to {agent_name}. Reason: {reason}"

    tool = StructuredTool.from_function(
        func=handoff_wrapper,
        name=f"transfer_to_{agent_name.lower()}",
        description=actual_description,
        args_schema=HandoffInput,
    )

    # 保存元数据
    tool._handoff_target = agent_name

    return tool


class HandoffAgent:
    """支持动态切换的 Agent

    Agent 可以主动将控制权切换给其他 Agent。

    示意图:
    ┌───────┐ handoff   ┌───────┐
    │ Alice │──────────→│  Bob  │
    └───────┘           └───────┘
        ↑                   │
        └───────────────────┘
             handoff
    """

    def __init__(
        self,
        name: str,
        llm_service: LLMService,
        tools: list = None,
        handoff_targets: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """初始化可切换 Agent

        Args:
            name: Agent 名称
            llm_service: LLM 服务
            tools: 工具列表
            handoff_targets: 可以切换到的目标 Agent 名称
            system_prompt: 系统提示词
        """
        self.name = name
        self._llm_service = llm_service
        self._tools = tools or []
        self._handoff_targets = handoff_targets or []
        self._system_prompt = system_prompt or f"You are {name}, a helpful assistant."
        self._handoff_tools: list = []

        # 创建切换工具
        for target in self._handoff_targets:
            handoff_tool = create_handoff_tool(target)
            self._handoff_tools.append(handoff_tool)

    async def _node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> Command:
        """Agent 节点"""
        from langchain_core.messages import SystemMessage

        # 准备消息
        messages = list(state["messages"])

        # 添加系统提示
        if not any(m.type == "system" for m in messages):
            messages.insert(0, SystemMessage(content=self._system_prompt))

        # 绑定所有工具（包括切换工具）
        all_tools = self._tools + self._handoff_tools
        llm_with_tools = self._llm_service.get_llm()
        if llm_with_tools and all_tools:
            llm_with_tools = llm_with_tools.bind_tools(all_tools)

        # 调用 LLM
        if llm_with_tools:
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self._llm_service.call(messages)

        # 检查是否是切换请求
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                if "transfer_to_" in tool_name:
                    # 执行切换
                    target = tool_name.replace("transfer_to_", "").capitalize()
                    logger.info("agent_initiated_handoff", from_agent=self.name, to=target)
                    return Command(goto=target, update={"messages": [response]})

        # 正常响应
        return Command(update={"messages": [response]}, goto=END)


def create_swarm(
    agents: list[HandoffAgent],
    default_agent: str,
) -> StateGraph:
    """创建 Agent Swarm

    Args:
        agents: HandoffAgent 列表
        default_agent: 默认激活的 Agent

    Returns:
        编译后的 StateGraph
    """
    builder = StateGraph(AgentState)

    # 添加所有 Agent 节点
    for agent in agents:
        builder.add_node(agent.name, agent._node)

    # 设置入口
    builder.set_entry_point(default_agent)

    return builder.compile()


# ============== 便捷函数 ==============

def create_multi_agent_system(
    mode: str = "router",
    llm_service: LLMService | None = None,
    **kwargs,
) -> StateGraph:
    """创建多 Agent 系统的便捷函数

    Args:
        mode: 多 Agent 模式 (router/supervisor/swarm)
        llm_service: LLM 服务
        **kwargs: 模式特定参数

    Returns:
        编译后的 StateGraph
    """
    if mode == "router":
        agents = kwargs.get("agents", {})
        return RouterAgent(llm_service, agents).compile()

    elif mode == "supervisor":
        workers = kwargs.get("workers", {})
        return SupervisorAgent(llm_service, workers).compile()

    elif mode == "swarm":
        agents = kwargs.get("agents", [])
        default_agent = kwargs.get("default_agent", agents[0].name if agents else "Agent")
        return create_swarm(agents, default_agent)

    else:
        raise ValueError(f"Unknown multi-agent mode: {mode}")
