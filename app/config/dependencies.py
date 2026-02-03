"""配置依赖注入模块

提供 FastAPI 依赖注入提供者，替代全局单例模式。
"""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.agent.memory.base import BaseLongTermMemory
    from app.agent.memory.context import ContextManager
    from app.agent.memory.manager import MemoryManager, MemoryManagerFactory

logger = get_logger(__name__)


# LLM Service


@lru_cache
def _get_llm_service_cached():  # -> LLMService (使用字符串避免循环导入)
    """获取 LLM 服务（缓存，单例）"""
    from app.llm import get_llm_service

    return get_llm_service()


async def get_llm_service_dep():  # -> AsyncIterator[LLMService]
    """LLM 服务依赖注入提供者"""
    llm_service = _get_llm_service_cached()
    try:
        yield llm_service
    finally:
        pass


# Memory Manager


async def get_memory_manager_dep(
    session_id: str,
    user_id: str | None = None,
    long_term_memory: "BaseLongTermMemory | None" = None,
) -> AsyncIterator["MemoryManager"]:
    """Memory Manager 依赖注入提供者"""
    memory_manager = MemoryManager(
        session_id=session_id,
        user_id=user_id,
        long_term_memory=long_term_memory,
    )
    try:
        yield memory_manager
    finally:
        await memory_manager.close()


# Memory Manager Factory


def get_memory_manager_factory_dep() -> "MemoryManagerFactory":
    """Memory Manager 工厂依赖注入提供者"""
    return MemoryManagerFactory


# Context Manager


def get_context_manager_dep() -> "ContextManager":
    """Context Manager 依赖注入提供者"""
    from app.agent.memory.context import get_context_manager

    return get_context_manager()


# Settings


def get_settings_dep() -> Settings:
    """配置依赖注入提供者"""
    return get_settings()


# Checkpointer


async def get_checkpointer_dep():
    """Checkpointer 依赖注入提供者"""
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
