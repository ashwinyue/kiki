"""Router Agent 模式

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

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from app.agent.schemas import RouteDecision
from app.agent.state import AgentState
from app.llm import LLMService
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认最大迭代次数
DEFAULT_MAX_ITERATIONS = 20


class RouterAgent:
    """路由 Agent

    根据用户意图将请求路由到不同的子 Agent。

    使用标准 add_conditional_edges 模式实现路由。
    包含迭代计数保护，防止无限循环。
    """

    # 特殊标记：表示路由完成
    ROUTE_DONE = "__route_done__"

    def __init__(
        self,
        llm_service: LLMService,
        agents: dict[str, Any],
        router_prompt: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """初始化路由 Agent

        Args:
            llm_service: LLM 服务
            agents: 子 Agent 字典 {name: AgentGraph}
            router_prompt: 路由提示词
            max_iterations: 最大迭代次数（防止路由循环）
        """
        self._llm_service = llm_service
        self._agents = agents
        self._agent_names = list(agents.keys())
        self._router_prompt = router_prompt or self._default_router_prompt()
        self._max_iterations = max_iterations
        self._graph: StateGraph | None = None

        # 构建结构化输出 LLM
        self._structured_llm = self._build_structured_llm()

    def _default_router_prompt(self) -> str:
        """默认路由提示词"""
        agent_list = ", ".join(self._agent_names)
        return f"""你是一个路由助手，负责将用户请求路由到合适的子 Agent。

可用的子 Agent:
{agent_list}

请分析用户意图，并选择最合适的 Agent 处理请求。
如果不确定，请选择第一个可用的 Agent。"""

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
        return ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_input}"),
            ]
        )

    async def _route_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """路由节点 - 决定使用哪个子 Agent

        使用 with_structured_output 获取类型安全的路由决策。
        将路由决策存储在状态中，由条件边函数读取。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含路由决策
        """
        from app.core.errors import classify_error

        last_message = state["messages"][-1]
        user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

        try:
            # 构建提示词
            prompt_template = self._build_router_prompt_template()

            # 使用 LCEL 链调用
            chain = prompt_template | self._structured_llm
            decision: RouteDecision = await chain.ainvoke(
                {
                    "system_prompt": self._router_prompt,
                    "user_input": user_input,
                }
            )

            # 验证目标 Agent 存在
            target_agent = decision.agent
            if target_agent not in self._agents:
                logger.warning("agent_not_found", target=target_agent, fallback="first")
                target_agent = self._agent_names[0] if self._agent_names else "general"

            logger.info(
                "routed_to",
                target=target_agent,
                reason=decision.reason,
                confidence=decision.confidence,
            )

            # 添加路由决策消息和路由目标
            decision_message = AIMessage(content=f"路由到 {target_agent}：{decision.reason}")

            return {
                "_next_agent": target_agent,
                "messages": [decision_message],
                "iteration_count": 1,  # 递增迭代计数器
            }

        except Exception as e:
            error_context = classify_error(e)
            logger.exception(
                "route_failed",
                error=str(e),
                category=error_context.category.value,
            )

            # 回退到第一个可用的 Agent
            fallback_agent = self._agent_names[0] if self._agent_names else "general"
            return {"_next_agent": fallback_agent}

    def _route_edge(self, state: AgentState) -> str:
        """条件边函数 - 根据路由决策决定下一个节点

        首先检查迭代次数，防止路由循环。

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        # 检查迭代次数，防止无限循环
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= self._max_iterations:
            logger.warning(
                "router_max_iterations_reached",
                iteration_count=iteration_count,
                max_iterations=self._max_iterations,
            )
            return END

        target = state.get("_next_agent", "")
        if target in self._agents:
            return target
        # 默认返回第一个可用的 Agent
        return self._agent_names[0] if self._agent_names else END

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

        # 添加条件边：从路由节点到各个子 Agent
        route_mapping = {name: name for name in self._agent_names}
        builder.add_conditional_edges("router", self._route_edge, route_mapping)

        # 每个子 Agent 完成后结束
        for name in self._agent_names:
            builder.add_edge(name, END)

        self._graph = builder.compile()
        return self._graph
