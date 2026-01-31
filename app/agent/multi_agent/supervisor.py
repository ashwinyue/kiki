"""Supervisor Agent 模式

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

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from app.agent.schemas import SupervisorDecision
from app.agent.state import AgentState
from app.llm import LLMService
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认最大迭代次数
DEFAULT_MAX_ITERATIONS = 20


class SupervisorAgent:
    """监督 Agent

    一个 Supervisor 管理多个 Worker Agent，协调它们完成任务。

    使用标准 add_conditional_edges 模式实现监督路由。
    包含迭代计数保护，防止无限循环。
    """

    # 特殊标记：表示任务完成
    TASK_DONE = "__done__"

    def __init__(
        self,
        llm_service: LLMService,
        workers: dict[str, Any],
        supervisor_prompt: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """初始化监督 Agent

        Args:
            llm_service: LLM 服务
            workers: Worker Agent 字典
            supervisor_prompt: 监督提示词
            max_iterations: 最大迭代次数（防止监督循环）
        """
        self._llm_service = llm_service
        self._workers = workers
        self._worker_names = list(workers.keys())
        self._supervisor_prompt = supervisor_prompt or self._default_supervisor_prompt()
        self._max_iterations = max_iterations
        self._graph: StateGraph | None = None

        # 构建结构化输出 LLM
        self._structured_llm = self._build_structured_llm()

    def _default_supervisor_prompt(self) -> str:
        """默认监督提示词"""
        worker_list = ", ".join(self._worker_names)
        return f"""你是一个监督者，负责协调以下 Worker Agent 完成任务:

Worker: {worker_list}

你的职责:
1. 分析任务需求
2. 将任务分配给合适的 Worker
3. 汇总 Worker 的结果
4. 决定任务是否完成或需要更多工作

当任务完成时，返回 next="{self.TASK_DONE}" 和 status="done"。"""

    def _build_structured_llm(self) -> BaseChatModel:
        """构建带结构化输出的 LLM"""
        llm = self._llm_service.get_llm()
        if llm is None:
            raise RuntimeError("LLM 未初始化")

        return llm.with_structured_output(SupervisorDecision)

    def _build_supervisor_prompt_template(self) -> ChatPromptTemplate:
        """构建监督提示词模板"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("system", "当前对话历史:\n{context}"),
            ]
        )

    async def _supervise_node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """监督节点 - 决定下一步

        使用 with_structured_output 获取类型安全的监督决策。
        将决策存储在状态中，由条件边函数读取。

        Args:
            state: 当前状态
            config: 运行配置

        Returns:
            状态更新，包含监督决策
        """
        from app.config.errors import classify_error

        # 构建上下文
        context = "\n".join(
            [f"- {m.type}: {str(m.content)[:100]}..." for m in state["messages"][-5:]]
        )

        try:
            # 构建提示词
            prompt_template = self._build_supervisor_prompt_template()

            # 使用 LCEL 链调用
            chain = prompt_template | self._structured_llm
            decision: SupervisorDecision = await chain.ainvoke(
                {
                    "system_prompt": self._supervisor_prompt,
                    "context": context,
                }
            )

            logger.info(
                "supervisor_decision",
                next=decision.next,
                status=decision.status,
            )

            # 决定下一步
            if (
                decision.status == "done"
                or decision.next == "END"
                or decision.next == self.TASK_DONE
            ):
                # 任务完成
                summary_message = AIMessage(content=decision.message or "任务已完成")
                return {
                    "_next_worker": self.TASK_DONE,
                    "messages": [summary_message],
                    "iteration_count": 1,  # 递增迭代计数器
                }

            # 路由到 Worker
            if decision.next in self._workers:
                worker_message = AIMessage(content=decision.message or f"分配给 {decision.next}")
                return {
                    "_next_worker": decision.next,
                    "messages": [worker_message],
                    "iteration_count": 1,  # 递增迭代计数器
                }

            # Worker 不存在，任务完成
            return {"_next_worker": self.TASK_DONE, "iteration_count": 1}

        except Exception as e:
            error_context = classify_error(e)
            logger.exception(
                "supervise_failed",
                error=str(e),
                category=error_context.category.value,
            )
            return {"_next_worker": self.TASK_DONE}

    def _supervise_edge(self, state: AgentState) -> str:
        """条件边函数 - 根据监督决策决定下一个节点

        首先检查迭代次数，防止监督循环。

        Args:
            state: 当前状态

        Returns:
            下一个节点名称或 END
        """
        # 检查迭代次数，防止无限循环
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= self._max_iterations:
            logger.warning(
                "supervisor_max_iterations_reached",
                iteration_count=iteration_count,
                max_iterations=self._max_iterations,
            )
            return END

        next_worker = state.get("_next_worker", "")
        if next_worker == self.TASK_DONE:
            return END
        if next_worker in self._workers:
            return next_worker
        # 默认返回第一个 Worker 或结束
        return self._worker_names[0] if self._worker_names else END

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

        # 添加条件边：从监督节点到各个 Worker 或 END
        route_mapping = {name: name for name in self._worker_names}
        route_mapping[self.TASK_DONE] = END
        builder.add_conditional_edges("supervisor", self._supervise_edge, route_mapping)

        # 每个 Worker 完成后返回监督
        for name in self._worker_names:
            builder.add_edge(name, "supervisor")

        self._graph = builder.compile()
        return self._graph
