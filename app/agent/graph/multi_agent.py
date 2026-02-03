"""Multi-Agent Graph Builder

使用 LangGraph 实现多种 Multi-Agent 模式：
- Supervisor Pattern：协调多个 worker agents
- Router Pattern：根据意图路由到不同 agent
- Hierarchical Pattern：分层的 agent 结构

完整支持调用链追踪和性能监控。
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agent.graph.checkpoint import get_checkpointer
from app.agent.state import MultiAgentState  # 从统一状态模块导入
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger
from app.repositories.agent_execution import (
    AgentExecutionTracker,
)

logger = get_logger(__name__)


@asynccontextmanager
async def agent_execution_context(
    session,
    session_id: str,
    thread_id: str,
    agent_id: str,
    agent_type: str,
    input_data: dict[str, Any] | None = None,
    parent_execution_id: UUID | None = None,
):
    """Agent 执行追踪上下文管理器

    自动记录 Agent 执行的开始和完成。

    Usage:
        async with agent_execution_context(...) as tracker:
            # 执行 agent 逻辑
            result = await run_agent()
            # 自动完成记录
            # (离开 context 时自动调用 complete_execution)
        pass
    """
    tracker = AgentExecutionTracker(session)

    execution = await tracker.start_execution(
        session_id=session_id,
        thread_id=thread_id,
        agent_id=agent_id,
        agent_type=agent_type,
        input_data=input_data or {},
        parent_execution_id=parent_execution_id,
    )

    try:
        yield tracker

        await tracker.complete_current_execution(
            output_data={"status": "success"},
            error_message=None,
        )

    except Exception as e:
        await tracker.complete_current_execution(
            output_data={"status": "error"},
            error_message=str(e),
        )
        raise


async def supervisor_node(
    state: MultiAgentState,
    config: dict[str, Any],
    allowed_workers: list[str],
    supervisor_prompt: str | None = None,
) -> Command:
    """Supervisor 节点：决定调用哪个 worker agent

    包含完整的调用链追踪。

    Args:
        state: 当前状态
        config: 配置（包含 session, thread_id 等）
        allowed_workers: 允许调用的 worker agent ID 列表
        supervisor_prompt: Supervisor 提示词

    Returns:
        Command 对象，指定下一个调用的 agent
    """
    messages = state.get("messages", [])
    if not messages:
        return Command(goto=END)

    last_message = messages[-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    # 简单路由逻辑
    next_agent = _route_by_keywords(user_input, allowed_workers)

    logger.info(
        "supervisor_routing",
        user_input=user_input[:100],
        next_agent=next_agent,
        allowed_workers=allowed_workers,
    )

    # TODO: 在这里可以添加 supervisor 执行追踪
    # async with agent_execution_context(...) as tracker:
    #     # 记录 supervisor 执行
    #     pass

    return Command(goto=(next_agent,), update={"next_agent": next_agent})


def _route_by_keywords(
    user_input: str,
    allowed_workers: list[str],
) -> str:
    """基于关键词的路由逻辑（示例实现）

    实际应用中应使用 LLM 进行意图分类。

    Args:
        user_input: 用户输入
        allowed_workers: 允许的 worker 列表

    Returns:
        选中的 agent ID
    """
    user_input_lower = user_input.lower()

    # 简单的关键词匹配规则
    routing_rules = {
        "search": ["搜索", "查找", "search", "google", "bing"],
        "code": ["代码", "编程", "运行", "执行", "code", "python", "javascript"],
        "data": ["数据", "分析", "统计", "data", "analyze"],
        "knowledge": ["知识", "文档", "资料", "knowledge", "document", "rag"],
    }

    # 检查每个规则
    for agent_id, keywords in routing_rules.items():
        if agent_id in allowed_workers:
            if any(keyword in user_input_lower for keyword in keywords):
                return agent_id

    # 默认返回第一个允许的 agent
    return allowed_workers[0] if allowed_workers else "rag-agent"


# ============== Worker Agent 包装器 ==============


def create_worker_node(agent_id: str, agent_config: dict[str, Any]):
    """创建 worker agent 节点函数（包含调用链追踪）

    Args:
        agent_id: Agent ID
        agent_config: Agent 配置

    Returns:
        节点函数
    """

    async def worker_node(state: MultiAgentState, config: dict[str, Any]) -> MultiAgentState:
        """Worker agent 执行节点（包含完整追踪）"""
        from sqlalchemy.ext.asyncio import AsyncSession

        from app.agent.chat_agent import ChatAgent

        logger.info("worker_agent_starting", agent_id=agent_id)

        # 获取 session（用于数据库操作）
        session = AsyncSession(config.get("bind"))

        try:
            # 从 state 中获取父执行 ID
            parent_execution_id = state.get("current_execution_id")

            # 获取用户消息
            messages = state.get("messages", [])
            if not messages:
                return state

            last_message = messages[-1]
            user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

            # 获取 session_id 和 thread_id
            configurable = config.get("configurable", {})
            session_id = configurable.get("thread_id", "default-thread")
            thread_id = session_id  # 可以复用 session_id 作为 thread_id

            # 创建追踪器并记录执行
            tracker = AgentExecutionTracker(session)

            execution = await tracker.start_execution(
                session_id=session_id,
                thread_id=thread_id,
                agent_id=agent_id,
                agent_type="worker",
                input_data={"user_input": user_input, "messages_count": len(messages)},
                parent_execution_id=parent_execution_id,
                meta_data={"agent_config": agent_config},  # 重命名避免 SQLAlchemy 保留字冲突
            )

            # 更新 state 中的当前执行 ID（供子 agent 使用）
            state["current_execution_id"] = execution.id

            # 创建 ChatAgent 实例
            agent = ChatAgent(
                system_prompt=agent_config.get("system_prompt"),
                tenant_id=config.get("tenant_id"),
            )

            # 执行 agent
            response = await agent.get_response(user_input, session_id)

            # 记录输出
            await tracker.complete_current_execution(
                output_data={
                    "messages_count": len(response),
                    "agent_id": agent_id,
                    "response_preview": str(response[-1])[:200] if response else None,
                },
                error_message=None,
            )

            # 更新状态
            output = {
                "messages": response,
                "agent_id": agent_id,
            }

            logger.info(
                "worker_agent_completed",
                execution_id=str(execution.id),
                agent_id=agent_id,
                response_length=len(response),
                duration_ms=execution.duration_ms,
            )

            return {
                "agent_outputs": {**state.get("agent_outputs", {}), agent_id: output},
                "current_agent_role": "worker",
            }

        except Exception as e:
            logger.error(
                "worker_agent_failed",
                agent_id=agent_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            # 记录失败
            if "current_execution_id" in state:
                tracker = AgentExecutionTracker(session)
                await tracker.complete_current_execution(
                    output_data={"error": str(e)},
                    error_message=str(e),
                )

            raise

        finally:
            await session.close()

    return worker_node


# ============== Graph 构建器 ==============


class MultiAgentGraphBuilder:
    """Multi-Agent Graph 构建器

    支持多种 Multi-Agent 模式：
    - Supervisor Pattern
    - Router Pattern
    - Hierarchical Pattern
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        workers: dict[str, dict[str, Any]] | None = None,
        supervisor_prompt: str | None = None,
    ):
        """初始化 Multi-Agent Graph 构建器

        Args:
            llm_service: LLM 服务实例
            workers: Worker agent 配置字典 {agent_id: agent_config}
            supervisor_prompt: Supervisor 提示词
        """
        self._llm_service = llm_service or get_llm_service()
        self._workers = workers or {}
        self._supervisor_prompt = supervisor_prompt

        logger.info(
            "multi_agent_builder_created",
            worker_count=len(self._workers),
        )

    async def build_supervisor_graph(
        self,
    ) -> CompiledStateGraph:
        """构建 Supervisor Pattern 图

        结构：
            START -> supervisor -> [worker_a, worker_b, worker_c]
            [worker_a, worker_b, worker_c] -> supervisor -> END

        Returns:
            编译后的 StateGraph
        """
        builder = StateGraph(MultiAgentState)

        # 添加 supervisor 节点
        builder.add_node(
            "supervisor",
            lambda state, config: supervisor_node(
                state,
                config,
                allowed_workers=list(self._workers.keys()),
                supervisor_prompt=self._supervisor_prompt,
            ),
        )

        # 添加 worker 节点
        for agent_id, agent_config in self._workers.items():
            builder.add_node(agent_id, create_worker_node(agent_id, agent_config))

        # 设置入口点
        builder.add_edge(START, "supervisor")

        # 添加条件边：supervisor -> workers
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next_agent") or END,
        )

        # 每个 worker 完成后回到 supervisor
        for agent_id in self._workers.keys():
            builder.add_edge(agent_id, "supervisor")

        logger.info("supervisor_graph_built", worker_count=len(self._workers))

        # 编译图
        checkpointer = await get_checkpointer(use_postgres=True)
        graph = builder.compile(checkpointer=checkpointer)

        return graph

    async def build_router_graph(
        self,
        routing_fn: callable,
    ) -> CompiledStateGraph:
        """构建 Router Pattern 图

        Args:
            routing_fn: 路由函数，接收 messages，返回 agent_id

        结构：
            START -> router -> [agent_a, agent_b, agent_c] -> END

        Returns:
            编译后的 StateGraph
        """
        builder = StateGraph(MultiAgentState)

        # 添加 router 节点
        async def router_node(state: MultiAgentState, config: dict[str, Any]):
            messages = state.get("messages", [])
            agent_id = routing_fn(messages)
            return Command(goto=(agent_id,))

        builder.add_node("router", router_node)

        # 添加 worker 节点
        for agent_id, agent_config in self._workers.items():
            builder.add_node(agent_id, create_worker_node(agent_id, agent_config))

        # 设置入口点
        builder.add_edge(START, "router")

        # 添加条件边：router -> workers
        builder.add_conditional_edges(
            "router",
            lambda state: state.get("next_agent") or END,
        )

        # 每个 worker 完成后结束
        for agent_id in self._workers.keys():
            builder.add_edge(agent_id, END)

        logger.info("router_graph_built", worker_count=len(self._workers))

        # 编译图
        checkpointer = await get_checkpointer(use_postgres=True)
        graph = builder.compile(checkpointer=checkpointer)

        return graph


# ============== 便捷函数 ==============


async def build_multi_agent_graph(
    graph_type: Literal["supervisor", "router"] = "supervisor",
    workers: dict[str, dict[str, Any]] | None = None,
    supervisor_prompt: str | None = None,
    routing_fn: Callable | None = None,  # 修复类型注解
) -> CompiledStateGraph:
    """构建 Multi-Agent 图（便捷函数）

    Args:
        graph_type: 图类型
        workers: Worker 配置字典
        supervisor_prompt: Supervisor 提示词（supervisor 模式）
        routing_fn: 路由函数（router 模式）

    Returns:
        编译后的 StateGraph

    Examples:
        ```python
        # Supervisor 模式
        workers = {
            "rag-agent": {"system_prompt": "你是知识库专家"},
            "search-agent": {"system_prompt": "你是搜索专家"},
        }
        graph = await build_multi_agent_graph("supervisor", workers)

        # 执行
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "搜索最新新闻"}]},
            {"configurable": {"thread_id": "session-123"}}
        )
        ```
    """
    builder = MultiAgentGraphBuilder(workers=workers, supervisor_prompt=supervisor_prompt)

    if graph_type == "supervisor":
        return await builder.build_supervisor_graph()
    elif graph_type == "router":
        if routing_fn is None:
            raise ValueError("routing_fn is required for router graph")
        return await builder.build_router_graph(routing_fn)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")


__all__ = [
    # 构建器
    "MultiAgentGraphBuilder",
    "build_multi_agent_graph",
    # 节点
    "supervisor_node",
    "create_worker_node",
    # 上下文管理器
    "agent_execution_context",
]
