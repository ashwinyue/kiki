"""Agent 管理类

提供完整的 LangGraph Agent 管理功能，包括图创建、响应获取、流式处理等。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

# PostgreSQL 检查点保存器（可选）
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _postgres_available = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore
    _postgres_available = False
from langgraph.types import RunnableConfig

# PostgreSQL 连接池（可选）
try:
    from psycopg_pool import AsyncConnectionPool
    _psycopg_pool_available = True
except (ImportError, OSError):
    AsyncConnectionPool = None  # type: ignore
    _psycopg_pool_available = False

from app.core.agent.graphs import ChatGraph
from app.core.agent.state import create_state_from_input
from app.core.config import get_settings
from app.core.llm import LLMService, get_llm_service
from app.core.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


class LangGraphAgent:
    """LangGraph Agent 管理类

    提供完整的 Agent 功能：
    - 图创建和编译
    - 同步/异步响应获取
    - 流式响应
    - 聊天历史管理
    - PostgreSQL 检查点持久化
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """初始化 Agent

        Args:
            llm_service: LLM 服务实例
            system_prompt: 系统提示词
            checkpointer: 检查点保存器
        """
        self._llm_service = llm_service or get_llm_service()
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._checkpointer = checkpointer
        self._graph: ChatGraph | None = None
        self._connection_pool: AsyncConnectionPool | None = None

        logger.info(
            "langgraph_agent_initialized",
            model=self._llm_service.current_model,
            has_checkpointer=checkpointer is not None,
        )

    def _default_system_prompt(self) -> str:
        """默认系统提示词

        Returns:
            系统提示词
        """
        return """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

    async def _get_postgres_checkpointer(self) -> AsyncPostgresSaver | None:
        """获取 PostgreSQL 检查点保存器

        Returns:
            AsyncPostgresSaver 实例或 None
        """
        if not _postgres_available or not _psycopg_pool_available:
            logger.debug("postgres_checkpointer_not_available")
            return None

        if self._checkpointer is not None:
            return self._checkpointer

        try:
            if self._connection_pool is None:

                # 从 database_url 解析连接信息
                db_url = settings.database_url
                if db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

                self._connection_pool = AsyncConnectionPool(
                    conninfo=db_url,
                    open=False,
                    max_size=settings.database_pool_size,
                    kwargs={"autocommit": True},
                )
                await self._connection_pool.open()
                logger.info("postgres_connection_pool_created")

            checkpointer = AsyncPostgresSaver(self._connection_pool)
            await checkpointer.setup()
            logger.info("postgres_checkpointer_initialized")
            return checkpointer

        except Exception as e:
            logger.warning("postgres_checkpointer_init_failed", error=str(e))
            return None

    def _get_graph(self) -> ChatGraph:
        """获取或创建图

        Returns:
            ChatGraph 实例
        """
        if self._graph is None:
            self._graph = ChatGraph(
                llm_service=self._llm_service,
                system_prompt=self._system_prompt,
            )
        return self._graph

    async def get_response(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
    ) -> list[BaseMessage]:
        """获取 Agent 响应

        Args:
            message: 用户消息
            session_id: 会话 ID（用于状态持久化）
            user_id: 用户 ID

        Returns:
            响应消息列表
        """
        graph = self._get_graph()

        # 准备输入
        input_data = create_state_from_input(
            input_text=message,
            user_id=user_id,
            session_id=session_id,
        )

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        # 获取检查点
        checkpointer = await self._get_postgres_checkpointer()
        if checkpointer:
            graph.compile(checkpointer=checkpointer)

        # 调用图
        logger.info("agent_invoke_start", session_id=session_id, user_id=user_id)
        result = await graph.ainvoke(input_data, config)
        logger.info("agent_invoke_complete", session_id=session_id)

        return result["messages"]

    async def get_stream_response(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
    ) -> AsyncIterator[str]:
        """获取流式响应

        Args:
            message: 用户消息
            session_id: 会话 ID
            user_id: 用户 ID

        Yields:
            响应内容片段
        """
        graph = self._get_graph()

        # 准备输入
        input_data = create_state_from_input(
            input_text=message,
            user_id=user_id,
            session_id=session_id,
        )

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        # 获取检查点
        checkpointer = await self._get_postgres_checkpointer()
        if checkpointer:
            graph.compile(checkpointer=checkpointer)

        # 流式调用
        logger.info("agent_stream_start", session_id=session_id)
        async for chunk in graph.astream(input_data, config, stream_mode="messages"):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
        logger.info("agent_stream_complete", session_id=session_id)

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
        graph = self._get_graph()

        config = RunnableConfig(
            configurable={"thread_id": session_id},
        )

        # 获取检查点
        checkpointer = await self._get_postgres_checkpointer()
        if checkpointer:
            graph.compile(checkpointer=checkpointer)

        state = await graph.aget_state(config)

        if state and state.values:
            return state.values.get("messages", [])

        return []

    async def clear_chat_history(self, session_id: str) -> None:
        """清除聊天历史

        Args:
            session_id: 会话 ID
        """
        try:
            checkpointer = await self._get_postgres_checkpointer()
            if checkpointer and self._connection_pool:
                async with self._connection_pool.connection() as conn:
                    # 删除检查点数据
                    await conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = %s",
                        (session_id,),
                    )
                    await conn.execute(
                        "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                        (session_id,),
                    )
                    logger.info("chat_history_cleared", session_id=session_id)
        except Exception as e:
            logger.error("clear_chat_history_failed", session_id=session_id, error=str(e))
            raise

    async def close(self) -> None:
        """关闭 Agent，释放资源"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("agent_closed")


# 全局 Agent 实例
_agent: LangGraphAgent | None = None


async def get_agent(
    system_prompt: str | None = None,
    use_postgres_checkpointer: bool = True,
) -> LangGraphAgent:
    """获取全局 Agent 实例（单例）

    Args:
        system_prompt: 自定义系统提示词
        use_postgres_checkpointer: 是否使用 PostgreSQL 检查点

    Returns:
        LangGraphAgent 实例
    """
    global _agent

    if _agent is None:
        checkpointer = None
        if use_postgres_checkpointer:
            # 延迟初始化检查点
            pass

        _agent = LangGraphAgent(system_prompt=system_prompt, checkpointer=checkpointer)

    return _agent


def create_agent(
    system_prompt: str | None = None,
    llm_service: LLMService | None = None,
) -> LangGraphAgent:
    """创建新的 Agent 实例

    Args:
        system_prompt: 系统提示词
        llm_service: LLM 服务实例

    Returns:
        LangGraphAgent 实例
    """
    return LangGraphAgent(
        llm_service=llm_service,
        system_prompt=system_prompt,
    )
