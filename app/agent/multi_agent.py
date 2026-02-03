"""Multi-Agent 实现

提供统一的多 Agent 入口，包括：
- SupervisorAgent: Supervisor 模式
- RouterAgent: Router 模式

设计原则:
    - 单一职责: 每个 Agent 类专注一种模式
    - 开闭原则: 易于扩展新的多 Agent 模式
    - 依赖倒置: 依赖 BaseAgent 抽象接口

使用示例:
    ```python
    from app.agent import SupervisorAgent

    # Supervisor 模式
    workers = {
        "search-agent": {"system_prompt": "你是搜索专家"},
        "rag-agent": {"system_prompt": "你是知识库专家"},
    }

    async with SupervisorAgent(workers=workers) as agent:
        response = await agent.get_response("搜索 AI 最新进展", session_id="session-123")

    # Router 模式
    async with RouterAgent(workers=workers, routing_fn=my_routing_fn) as agent:
        response = await agent.get_response("查询知识库", session_id="session-123")
    ```
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RunnableConfig

from app.agent.base import BaseAgent
from app.agent.graph.checkpoint import get_checkpointer
from app.agent.graph.multi_agent import (
    MultiAgentGraphBuilder,
    supervisor_node,
    create_worker_node,
)
from app.agent.state import MultiAgentState
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== Multi-Agent 基类 ==============


class MultiAgent(BaseAgent):
    """Multi-Agent 基类

    提供多 Agent 的通用功能。

    Attributes:
        _workers: Worker Agent 配置字典 {agent_id: agent_config}
        _llm_service: LLM 服务实例
        _checkpointer: 检查点保存器
        _tenant_id: 租户 ID
        _graph: 编译后的图（延迟初始化）
    """

    def __init__(
        self,
        workers: dict[str, dict[str, Any]],
        llm_service: LLMService | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tenant_id: int | None = None,
    ) -> None:
        """初始化 Multi-Agent

        Args:
            workers: Worker Agent 配置字典
                ```python
                {
                    "agent-id": {
                        "system_prompt": "系统提示词",
                        "tools": [tool1, tool2],  # 可选
                        "temperature": 0.7,  # 可选
                    }
                }
                ```
            llm_service: LLM 服务实例
            checkpointer: 检查点保存器
            tenant_id: 租户 ID
        """
        self._workers = workers
        self._llm_service = llm_service or get_llm_service()
        self._checkpointer = checkpointer
        self._tenant_id = tenant_id
        self._graph: CompiledStateGraph | None = None

        logger.info(
            "multi_agent_created",
            worker_count=len(workers),
            has_checkpointer=checkpointer is not None,
            tenant_id=tenant_id,
        )

    async def _ensure_graph(self) -> CompiledStateGraph:
        """确保图已编译（子类必须实现）"""
        raise NotImplementedError("Subclass must implement _ensure_graph")

    async def get_response(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        tenant_id: int | None = None,
    ) -> list[BaseMessage]:
        """获取 Agent 响应

        Args:
            message: 用户消息
            session_id: 会话 ID
            user_id: 用户 ID（未使用，保持接口兼容）
            tenant_id: 租户 ID（未使用，保持接口兼容）

        Returns:
            消息列表
        """
        graph = await self._ensure_graph()

        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "tenant_id": tenant_id or self._tenant_id,
                "user_id": user_id,
            },
        }

        logger.debug(
            "multi_agent_invoking",
            session_id=session_id,
            message_length=len(message),
        )

        try:
            state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "iteration_count": 0,
                    "max_iterations": 10,
                    "agent_outputs": {},
                    "agent_history": [],
                },
                config,
            )

            messages = state.get("messages", [])
            logger.info(
                "multi_agent_responded",
                session_id=session_id,
                message_count=len(messages),
                agent_outputs_count=len(state.get("agent_outputs", {})),
            )

            return messages

        except Exception as e:
            logger.exception("multi_agent_failed", session_id=session_id, error=str(e))
            raise RuntimeError(f"Multi-Agent 调用失败: {e}") from e

    async def astream(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        tenant_id: int | None = None,
    ) -> AsyncIterator[BaseMessage]:
        """流式获取 Agent 响应

        Args:
            message: 用户消息
            session_id: 会话 ID
            user_id: 用户 ID（未使用）
            tenant_id: 租户 ID（未使用）

        Yields:
            消息（逐个产出）
        """
        graph = await self._ensure_graph()

        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "metadata": {
                "tenant_id": tenant_id or self._tenant_id,
                "user_id": user_id,
            },
        }

        logger.debug(
            "multi_agent_streaming",
            session_id=session_id,
            message_length=len(message),
        )

        try:
            async for event in graph.astream(
                {
                    "messages": [HumanMessage(content=message)],
                    "iteration_count": 0,
                    "max_iterations": 10,
                    "agent_outputs": {},
                    "agent_history": [],
                },
                config,
                stream_mode="messages",
            ):
                if isinstance(event, tuple) and len(event) == 2:
                    msg, metadata = event
                    if isinstance(msg, BaseMessage):
                        yield msg
                elif isinstance(event, BaseMessage):
                    yield event

        except Exception as e:
            logger.exception("multi_agent_stream_failed", session_id=session_id, error=str(e))
            raise RuntimeError(f"Multi-Agent 流式调用失败: {e}") from e

    async def close(self) -> None:
        """关闭 Agent，释放资源"""
        logger.debug("multi_agent_closed")


# ============== Supervisor Agent ==============


class SupervisorAgent(MultiAgent):
    """Supervisor 模式 Agent

    Supervisor 负责任务分解和路由，协调多个 Worker Agent 协作。

    图结构：
        START -> Supervisor -> [Worker_A, Worker_B, Worker_C]
        [Worker_A, Worker_B, Worker_C] -> Supervisor -> END

    优点:
        - 中央协调：Supervisor 统一决策
        - 灵活路由：根据任务动态选择 Worker
        - 迭代优化：支持多轮协作

    适用场景:
        - 需要任务分解的复杂场景
        - 需要多轮协作的任务
        - 需要中央协调控制

    示例:
        ```python
        workers = {
            "search-agent": {"system_prompt": "你是搜索专家"},
            "rag-agent": {"system_prompt": "你是知识库专家"},
            "code-agent": {"system_prompt": "你是代码专家"},
        }

        async with SupervisorAgent(workers=workers) as agent:
            response = await agent.get_response(
                "搜索 AI 最新进展并生成代码示例",
                session_id="session-123"
            )
        ```
    """

    def __init__(
        self,
        workers: dict[str, dict[str, Any]],
        supervisor_prompt: str | None = None,
        llm_service: LLMService | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tenant_id: int | None = None,
    ) -> None:
        """初始化 Supervisor Agent

        Args:
            workers: Worker Agent 配置字典
            supervisor_prompt: Supervisor 的提示词
                如果不提供，使用默认提示词
            llm_service: LLM 服务实例
            checkpointer: 检查点保存器
            tenant_id: 租户 ID
        """
        super().__init__(workers, llm_service, checkpointer, tenant_id)
        self._supervisor_prompt = supervisor_prompt

        logger.info(
            "supervisor_agent_created",
            worker_count=len(workers),
            has_custom_prompt=supervisor_prompt is not None,
        )

    async def _ensure_graph(self) -> CompiledStateGraph:
        """确保 Supervisor 图已编译"""
        if self._graph is None:
            logger.debug("compiling_supervisor_graph")

            builder = MultiAgentGraphBuilder(
                workers=self._workers,
                supervisor_prompt=self._supervisor_prompt,
            )

            self._graph = await builder.build_supervisor_graph()

            logger.info("supervisor_graph_compiled")
        return self._graph


# ============== Router Agent ==============


class RouterAgent(MultiAgent):
    """Router 模式 Agent

    Router 根据用户输入路由到合适的 Worker Agent，单次执行。

    图结构：
        START -> Router -> [Worker_A, Worker_B, Worker_C] -> END

    优点:
        - 快速路由：直接路由到目标 Agent
        - 无迭代：减少不必要的轮次
        - 简单高效：适合明确意图的场景

    适用场景:
        - 意图明确的任务
        - 单次执行即可完成
        - 不需要协作的场景

    示例:
        ```python
        def my_routing_fn(messages: list[BaseMessage]) -> str:
            # 自定义路由逻辑
            last_msg = messages[-1]
            if "搜索" in last_msg.content:
                return "search-agent"
            elif "知识库" in last_msg.content:
                return "rag-agent"
            return "default-agent"

        workers = {
            "search-agent": {"system_prompt": "你是搜索专家"},
            "rag-agent": {"system_prompt": "你是知识库专家"},
        }

        async with RouterAgent(workers=workers, routing_fn=my_routing_fn) as agent:
            response = await agent.get_response("搜索 AI 新闻", session_id="session-123")
        ```
    """

    def __init__(
        self,
        workers: dict[str, dict[str, Any]],
        routing_fn: Callable[[list[BaseMessage]], str],
        llm_service: LLMService | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tenant_id: int | None = None,
    ) -> None:
        """初始化 Router Agent

        Args:
            workers: Worker Agent 配置字典
            routing_fn: 路由函数，接收消息列表，返回 Agent ID
                ```python
                def routing_fn(messages: list[BaseMessage]) -> str:
                    # 返回应该调用的 agent_id
                    return "search-agent"
                ```
            llm_service: LLM 服务实例
            checkpointer: 检查点保存器
            tenant_id: 租户 ID
        """
        super().__init__(workers, llm_service, checkpointer, tenant_id)
        self._routing_fn = routing_fn

        logger.info(
            "router_agent_created",
            worker_count=len(workers),
        )

    async def _ensure_graph(self) -> CompiledStateGraph:
        """确保 Router 图已编译"""
        if self._graph is None:
            logger.debug("compiling_router_graph")

            builder = MultiAgentGraphBuilder(workers=self._workers)
            self._graph = await builder.build_router_graph(self._routing_fn)

            logger.info("router_graph_compiled")
        return self._graph


__all__ = [
    "MultiAgent",
    "SupervisorAgent",
    "RouterAgent",
]
