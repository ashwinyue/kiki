"""依赖注入模块

提供 FastAPI 依赖注入提供者，替代全局单例模式。
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config.settings import Settings, get_settings
from app.llm import LLMService
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.agent import LangGraphAgent
    from app.agent.memory.base import BaseLongTermMemory
    from app.agent.memory.context import ContextManager
    from app.agent.memory.manager import MemoryManager, MemoryManagerFactory

logger = get_logger(__name__)


# ============== LLM Service ==============


@lru_cache
def _get_llm_service_cached() -> LLMService:
    """获取 LLM 服务（缓存，单例）"""
    from app.llm import get_llm_service

    return get_llm_service()


async def get_llm_service_dep() -> AsyncIterator[LLMService]:
    """LLM 服务依赖注入提供者

    Returns:
        LLMService 实例
    """
    llm_service = _get_llm_service_cached()
    try:
        yield llm_service
    finally:
        # LLM 服务是无状态的，不需要清理
        pass


# ============== Agent ==============


class AgentContainer:
    """Agent 容器

    管理 Agent 实例的生命周期，支持多租户隔离。
    """

    def __init__(self) -> None:
        self._agents: dict[str, "LangGraphAgent"] = {}
        self._default_agent: "LangGraphAgent | None" = None

    async def get_agent(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> "LangGraphAgent":
        """获取 Agent 实例

        Args:
            session_id: 会话 ID（用于多租户隔离）
            user_id: 用户 ID

        Returns:
            LangGraphAgent 实例
        """
        # 如果没有会话 ID，返回默认 Agent
        if not session_id:
            if self._default_agent is None:
                from app.agent import create_agent

                self._default_agent = create_agent()
            return self._default_agent

        # 为每个会话创建独立的 Agent（支持多租户隔离）
        if session_id not in self._agents:
            from app.agent import create_agent

            self._agents[session_id] = create_agent()
            logger.debug("agent_created_for_session", session_id=session_id)

        return self._agents[session_id]

    async def close_session(self, session_id: str) -> None:
        """关闭会话的 Agent

        Args:
            session_id: 会话 ID
        """
        if session_id in self._agents:
            await self._agents[session_id].close()
            del self._agents[session_id]
            logger.debug("agent_closed_for_session", session_id=session_id)

    async def close_all(self) -> None:
        """关闭所有 Agent"""
        for agent in self._agents.values():
            await agent.close()
        self._agents.clear()
        if self._default_agent:
            await self._default_agent.close()
            self._default_agent = None


# 全局容器
_agent_container = AgentContainer()


async def get_agent_dep(
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncIterator["LangGraphAgent"]:
    """Agent 依赖注入提供者

    Args:
        session_id: 会话 ID
        user_id: 用户 ID

    Returns:
        LangGraphAgent 实例

    Examples:
        ```python
        @app.post("/chat")
        async def chat(
            agent: Annotated[LangGraphAgent, Depends(get_agent_dep)],
            message: str,
        ):
            # ...
        ```
    """
    agent = await _agent_container.get_agent(session_id, user_id)
    try:
        yield agent
    except Exception:
        logger.error("agent_dependency_error", session_id=session_id)
        raise


# ============== Memory Manager ==============


async def get_memory_manager_dep(
    session_id: str,
    user_id: str | None = None,
    long_term_memory: "BaseLongTermMemory | None" = None,
) -> AsyncIterator["MemoryManager"]:
    """Memory Manager 依赖注入提供者

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        long_term_memory: 长期记忆实例（通过依赖注入）

    Returns:
        MemoryManager 实例

    Examples:
        ```python
        @app.post("/chat")
        async def chat(
            memory: Annotated[MemoryManager, Depends(get_memory_manager_dep)],
            message: str,
        ):
            # ...
        ```
    """
    memory_manager = MemoryManager(
        session_id=session_id,
        user_id=user_id,
        long_term_memory=long_term_memory,
    )
    try:
        yield memory_manager
    finally:
        await memory_manager.close()


# ============== Memory Manager Factory ==============


def get_memory_manager_factory_dep() -> "MemoryManagerFactory":
    """Memory Manager 工厂依赖注入提供者

    Returns:
        MemoryManagerFactory 实例

    Examples:
        ```python
        @app.post("/chat")
        async def chat(
            factory: Annotated[MemoryManagerFactory, Depends(get_memory_manager_factory_dep)],
            message: str,
        ):
            memory = factory.create(session_id="test-123", user_id="user-456")
            # ...
        ```
    """
    return MemoryManagerFactory


# ============== Context Manager ==============


def get_context_manager_dep() -> "ContextManager":
    """Context Manager 依赖注入提供者

    Returns:
        ContextManager 实例
    """
    from app.agent.memory.context import get_context_manager

    return get_context_manager()


# ============== Settings ==============


def get_settings_dep() -> Settings:
    """配置依赖注入提供者

    Returns:
        Settings 实例
    """
    return get_settings()


# ============== Checkpointer ==============


async def get_checkpointer_dep():
    """Checkpointer 依赖注入提供者

    Returns:
        检查点保存器实例或 None
    """
    from app.config.settings import get_settings

    settings = get_settings()

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        db_url = settings.database_url
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        pool = AsyncConnectionPool(
            conninfo=db_url,
            open=False,
            max_size=settings.database_pool_size,
            kwargs={"autocommit": True},
        )
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        logger.info("checkpointer_created")

        try:
            yield checkpointer
        finally:
            await pool.close()

    except ImportError:
        logger.warning("postgres_checkpointer_not_available")
        yield None
    except Exception as e:
        logger.error("checkpointer_init_failed", error=str(e))
        yield None
