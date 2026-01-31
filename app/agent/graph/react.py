"""ReAct Agent 模块

基于 LangGraph 的 create_react_agent 提供快速开发选项。

ReAct (Reasoning + Acting) 模式是一种经典的 Agent 模式，
LLM 通过推理决定采取什么行动，然后执行行动并观察结果。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent as langgraph_create_react_agent
from langgraph.types import RunnableConfig

from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)


# PostgreSQL 检查点保存器（可选）
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _postgres_available = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore
    _postgres_available = False

# PostgreSQL 连接池（可选）
try:
    from psycopg_pool import AsyncConnectionPool

    _psycopg_pool_available = True
except (ImportError, OSError):
    AsyncConnectionPool = None  # type: ignore
    _psycopg_pool_available = False

# 延迟导入配置（需要避免循环依赖和初始化顺序问题）
# noqa: E402 - 必须在可选依赖检查之后导入
from app.config.settings import get_settings  # noqa: E402

settings = get_settings()


class ReactAgent:
    """ReAct Agent 封装类

    提供与 LangGraphAgent 相同的接口，但使用 LangGraph 内置的 create_react_agent。

    优点:
    - 开箱即用的 ReAct 模式
    - 自动处理工具调用循环
    - 更少的代码

    适用场景:
    - 快速原型开发
    - 简单的 Agent 应用
    - 不需要自定义状态管理

    生命周期管理:
        支持异步上下文管理器，确保资源正确释放：

        ```python
        async with ReactAgent(tools=[my_tool]) as agent:
            response = await agent.get_response("...", session_id="...")
        # 连接池自动关闭
        ```

        或者手动调用 close():

        ```python
        agent = ReactAgent(tools=[my_tool])
        try:
            response = await agent.get_response("...", session_id="...")
        finally:
            await agent.close()
        ```
    """

    # 类级别的连接池缓存（避免重复创建）
    _shared_connection_pool: AsyncConnectionPool | None = None
    _pool_ref_count: int = 0

    def __init__(
        self,
        llm_service: LLMService | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """初始化 ReAct Agent

        Args:
            llm_service: LLM 服务实例
            tools: 工具列表
            system_prompt: 系统提示词
            checkpointer: 检查点保存器（如果提供，不会自动创建 PostgreSQL checkpointer）
        """
        self._llm_service = llm_service or get_llm_service()
        self._tools = tools or []
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._checkpointer = checkpointer
        self._graph: CompiledStateGraph | None = None
        self._owns_pool: bool = False  # 是否拥有连接池的所有权

        logger.info(
            "react_agent_initialized",
            model=self._llm_service.current_model,
            tool_count=len(self._tools),
            has_checkpointer=checkpointer is not None,
        )

    async def __aenter__(self) -> "ReactAgent":
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口，确保资源释放"""
        await self.close()

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

    async def _get_postgres_checkpointer(self) -> AsyncPostgresSaver | None:
        """获取 PostgreSQL 检查点保存器

        使用类级别的共享连接池，避免重复创建。

        Returns:
            AsyncPostgresSaver 实例或 None
        """
        if not _postgres_available or not _psycopg_pool_available:
            logger.debug("postgres_checkpointer_not_available")
            return None

        if self._checkpointer is not None:
            return self._checkpointer

        try:
            # 使用类级别的共享连接池
            if ReactAgent._shared_connection_pool is None:
                # 从 database_url 解析连接信息
                db_url = settings.database_url
                if db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

                ReactAgent._shared_connection_pool = AsyncConnectionPool(
                    conninfo=db_url,
                    open=False,
                    max_size=settings.database_pool_size,
                    kwargs={"autocommit": True},
                )
                await ReactAgent._shared_connection_pool.open()
                logger.info("postgres_shared_connection_pool_created")

            ReactAgent._pool_ref_count += 1
            self._owns_pool = True

            checkpointer = AsyncPostgresSaver(ReactAgent._shared_connection_pool)
            await checkpointer.setup()
            logger.info(
                "postgres_checkpointer_initialized", pool_ref_count=ReactAgent._pool_ref_count
            )
            return checkpointer

        except Exception as e:
            logger.warning("postgres_checkpointer_init_failed", error=str(e))
            return None

    def _get_graph(self) -> CompiledStateGraph:
        """获取或创建图

        Returns:
            CompiledStateGraph 实例
        """
        if self._graph is None:
            llm = self._llm_service.get_llm()
            if llm is None:
                raise RuntimeError("LLM 未初始化")

            self._graph = langgraph_create_react_agent(
                model=llm,
                tools=self._tools,
                prompt=self._system_prompt,
            )
            logger.info("react_graph_created")

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
        from langchain_core.messages import HumanMessage

        graph = self._get_graph()

        # 准备输入
        input_data = {"messages": [HumanMessage(content=message)]}

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        # 调用图
        logger.info("react_agent_invoke_start", session_id=session_id, user_id=user_id)
        result = await graph.ainvoke(input_data, config)
        logger.info("react_agent_invoke_complete", session_id=session_id)

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
        from langchain_core.messages import HumanMessage

        graph = self._get_graph()

        # 准备输入
        input_data = {"messages": [HumanMessage(content=message)]}

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        # 流式调用
        logger.info("react_agent_stream_start", session_id=session_id)
        async for chunk in graph.astream(input_data, config, stream_mode="messages"):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
        logger.info("react_agent_stream_complete", session_id=session_id)

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
            if self._connection_pool:
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
        """关闭 Agent，释放资源

        使用共享连接池时，只有当所有实例都关闭后才真正关闭连接池。
        """
        if self._owns_pool and ReactAgent._shared_connection_pool is not None:
            ReactAgent._pool_ref_count -= 1
            logger.info("react_agent_pool_ref_decremented", ref_count=ReactAgent._pool_ref_count)

            # 只有当没有实例引用时才关闭连接池
            if ReactAgent._pool_ref_count <= 0:
                await ReactAgent._shared_connection_pool.close()
                ReactAgent._shared_connection_pool = None
                ReactAgent._pool_ref_count = 0
                logger.info("postgres_shared_connection_pool_closed")

            self._owns_pool = False

        logger.info("react_agent_closed")

    @classmethod
    async def shutdown_shared_pool(cls) -> None:
        """关闭类级别的共享连接池

        通常在应用关闭时调用。

        Example:
            ```python
            await ReactAgent.shutdown_shared_pool()
            ```
        """
        if cls._shared_connection_pool is not None:
            await cls._shared_connection_pool.close()
            cls._shared_connection_pool = None
            cls._pool_ref_count = 0
            logger.info("postgres_shared_connection_pool_shutdown")


# ============== 便捷函数 ==============


def create_react_agent(
    llm_service: LLMService | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> ReactAgent:
    """创建 ReAct Agent 实例

    这是一个快速创建 Agent 的便捷函数，适合简单场景。

    Args:
        llm_service: LLM 服务实例
        tools: 工具列表
        system_prompt: 系统提示词
        checkpointer: 检查点保存器

    Returns:
        ReactAgent 实例

    Examples:
        ```python
        from langchain_core.tools import tool
        from app.agent.graph import create_react_agent

        @tool
        async def get_weather(location: str) -> str:
            \"\"\"获取指定位置的天气\"\"\"
            return f"{location} 今天晴天，25°C"

        agent = create_react_agent(
            tools=[get_weather],
            system_prompt="你是一个天气助手",
        )

        response = await agent.get_response(
            message="北京今天天气怎么样？",
            session_id="session-123",
        )
        ```
    """
    return ReactAgent(
        llm_service=llm_service,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


__all__ = [
    "ReactAgent",
    "create_react_agent",
]
