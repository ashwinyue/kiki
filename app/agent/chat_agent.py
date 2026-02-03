"""Chat Agent 实现

使用 compile_chat_graph 创建的标准对话 Agent。
适合简单对话场景，不需要工具调用或需要自定义控制。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.agent.base import BaseAgent
from app.agent.graph.builder import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SYSTEM_PROMPT,
    compile_chat_graph,
)
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ChatAgent(BaseAgent):
    """Chat Agent（使用 compile_chat_graph）

    适合场景:
        - 简单对话
        - 不需要工具调用
        - 需要自定义控制

    优点:
        - 使用标准 LangGraph 模式
        - 工具在编译时绑定
        - 支持状态持久化

    示例:
        ```python
        # 基础使用
        agent = ChatAgent(system_prompt="你是一个有用的助手")
        response = await agent.get_response("你好", session_id="session-123")

        # 使用异步上下文管理器（推荐）
        async with ChatAgent() as agent:
            response = await agent.get_response("你好", session_id="session-123")
        ```
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tenant_id: int | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """初始化 Chat Agent

        Args:
            llm_service: LLM 服务实例（默认使用全局实例）
            system_prompt: 系统提示词（默认使用 DEFAULT_SYSTEM_PROMPT）
            checkpointer: 检查点保存器（用于状态持久化）
            tenant_id: 租户 ID
            max_iterations: 最大迭代次数
        """
        self._llm_service = llm_service or get_llm_service()
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._checkpointer = checkpointer
        self._tenant_id = tenant_id
        self._max_iterations = max_iterations
        self._graph: CompiledStateGraph | None = None

        logger.info(
            "chat_agent_created",
            model=self._llm_service.current_model,
            has_checkpointer=checkpointer is not None,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
        )

    async def _ensure_graph(self) -> CompiledStateGraph:
        """确保图已编译

        Returns:
            编译后的 StateGraph
        """
        if self._graph is None:
            logger.debug("compiling_chat_graph")
            self._graph = compile_chat_graph(
                llm_service=self._llm_service,
                system_prompt=self._system_prompt,
                checkpointer=self._checkpointer,
                tenant_id=self._tenant_id,
                max_iterations=self._max_iterations,
            )
            logger.info("chat_graph_compiled")
        return self._graph

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
        }

        logger.debug(
            "chat_agent_invoking",
            session_id=session_id,
            message_length=len(message),
        )

        try:
            state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=message)],
                },
                config,
            )

            messages = state.get("messages", [])
            logger.info(
                "chat_agent_responded",
                session_id=session_id,
                message_count=len(messages),
            )

            return messages

        except Exception as e:
            logger.exception("chat_agent_failed", session_id=session_id, error=str(e))
            raise RuntimeError(f"Chat Agent 调用失败: {e}") from e

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
        }

        logger.debug(
            "chat_agent_streaming",
            session_id=session_id,
            message_length=len(message),
        )

        try:
            async for event in graph.astream(
                {"messages": [HumanMessage(content=message)]},
                config,
                stream_mode="messages",
            ):
                # event 是 (message, metadata) 元组
                if isinstance(event, tuple) and len(event) == 2:
                    msg, metadata = event
                    if isinstance(msg, BaseMessage):
                        yield msg
                elif isinstance(event, BaseMessage):
                    yield event

        except Exception as e:
            logger.exception("chat_agent_stream_failed", session_id=session_id, error=str(e))
            raise RuntimeError(f"Chat Agent 流式调用失败: {e}") from e

    async def close(self) -> None:
        """关闭 Agent，释放资源"""
        if hasattr(self, "_connection_pool") and self._connection_pool:
            await self._connection_pool.close()
            logger.debug("chat_agent_connection_pool_closed")
        logger.debug("chat_agent_closed")

    async def get_chat_history(
        self,
        session_id: str,
    ) -> list[BaseMessage]:
        """获取聊天历史

        Args:
            session_id: 会话 ID

        Returns:
            历史消息列表
        """
        if self._checkpointer:
            graph = await self._ensure_graph()
            config: RunnableConfig = {
                "configurable": {"thread_id": session_id},
            }
            state = await graph.aget_state(config)
            if state and state.values:
                return state.values.get("messages", [])

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            from app.infra.database import session_scope
            from app.repositories.message import MessageRepository

            role_map = {
                "user": HumanMessage,
                "human": HumanMessage,
                "assistant": AIMessage,
                "ai": AIMessage,
                "system": SystemMessage,
            }

            async with session_scope() as session:
                repo = MessageRepository(session)
                db_messages = await repo.list_by_session_asc(session_id, limit=100)
                if db_messages:
                    return [
                        role_map.get(msg.role, HumanMessage)(content=msg.content)
                        for msg in db_messages
                    ]
        except Exception as e:
            logger.debug(
                "db_history_load_failed",
                session_id=session_id,
                error=str(e),
            )
        return []

    async def clear_chat_history(self, session_id: str) -> None:
        """清除聊天历史

        Args:
            session_id: 会话 ID
        """
        try:
            try:
                from app.infra.database import session_scope
                from app.repositories.message import MessageRepository

                async with session_scope() as session:
                    repo = MessageRepository(session)
                    await repo.delete_by_session(session_id)
            except Exception as e:
                logger.debug(
                    "message_db_clear_failed",
                    session_id=session_id,
                    error=str(e),
                )

            # 清理 checkpoint（PostgreSQL checkpointer 有 conn 属性）
            if self._checkpointer and hasattr(self._checkpointer, "conn"):
                async with self._checkpointer.conn.connection() as conn:
                    await conn.execute(
                        "DELETE FROM checks WHERE thread_id = $1",
                        session_id,
                    )
                    await conn.execute(
                        "DELETE FROM checkpoints_blobs WHERE thread_id = $1",
                        session_id,
                    )
                    await conn.execute(
                        "DELETE FROM checkpoint_writes WHERE thread_id = $1",
                        session_id,
                    )
                    logger.info("chat_history_cleared", session_id=session_id)
        except Exception as e:
            logger.error("clear_chat_history_failed", session_id=session_id, error=str(e))
            raise

    async def get_compiled_graph(self) -> CompiledStateGraph:
        """获取已编译图

        Returns:
            编译后的 StateGraph
        """
        return await self._ensure_graph()


__all__ = ["ChatAgent"]
