"""数据库服务模块

提供数据库连接池、会话管理和事务处理。
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel, create_engine

from app.config.settings import get_settings
from app.observability.logging import get_logger
from app.repositories import (
    MessageRepository,
    MCPServiceRepository,
    SessionRepository,
    ThreadRepository,
    UserRepository,
)

logger = get_logger(__name__)

settings = get_settings()

# 同步引擎（用于迁移）
_sync_engine = None

# 异步引擎
_async_engine: AsyncEngine | None = None

# 会话工厂
_session_factory = None


def get_sync_engine():
    """获取同步数据库引擎

    Returns:
        同步引擎实例
    """
    global _sync_engine
    if _sync_engine is None:
        # 转换 asyncpg 连接字符串为 psycopg
        db_url = settings.database_url
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        # SQLite 不支持 pool_size 和 max_overflow
        if db_url.startswith("sqlite"):
            engine_args = {"echo": settings.database_echo}
        else:
            engine_args = {
                "echo": settings.database_echo,
                "pool_size": 20,
                "max_overflow": 10,
            }

        _sync_engine = create_engine(db_url, **engine_args)
        logger.info("sync_db_engine_created")
    return _sync_engine


def get_async_engine() -> AsyncEngine:
    """获取异步数据库引擎

    Returns:
        异步引擎实例
    """
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            pool_pre_ping=True,  # 连接前检查有效性
        )
        logger.info("async_db_engine_created", pool_size=settings.database_pool_size)
    return _async_engine


def _get_session_factory():
    """获取会话工厂"""
    global _session_factory
    if _session_factory is None:
        engine = get_async_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话（依赖注入）

    Yields:
        异步会话实例
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession]:
    """会话作用域上下文管理器

    Yields:
        异步会话实例
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def init_db():
    """初始化数据库（创建表）"""

    engine = get_sync_engine()
    SQLModel.metadata.create_all(engine)
    logger.info("database_tables_created")


# ============== 事务辅助方法 ==============


async def transaction(
    func: Callable[[AsyncSession], Any],
) -> Any:
    """执行事务

    Args:
        func: 要在事务中执行的函数

    Returns:
        函数的返回值

    Raises:
        Exception: 事务执行失败时回滚并抛出异常
    """
    async with session_scope() as session:
        try:
            result = await func(session)
            await session.commit()
            return result
        except Exception:
            await session.rollback()
            raise


# ============== 仓储工厂方法 ==============


def user_repository(session: AsyncSession) -> UserRepository:
    """创建用户仓储

    Args:
        session: 数据库会话

    Returns:
        UserRepository 实例
    """
    return UserRepository(session)


def session_repository(session: AsyncSession) -> SessionRepository:
    """创建会话仓储

    Args:
        session: 数据库会话

    Returns:
        SessionRepository 实例
    """
    return SessionRepository(session)


def thread_repository(session: AsyncSession) -> ThreadRepository:
    """创建线程仓储

    Args:
        session: 数据库会话

    Returns:
        ThreadRepository 实例
    """
    return ThreadRepository(session)


def mcp_service_repository(session: AsyncSession) -> MCPServiceRepository:
    """创建 MCP 服务仓储

    Args:
        session: 数据库会话

    Returns:
        MCPServiceRepository 实例
    """
    return MCPServiceRepository(session)


def message_repository(session: AsyncSession) -> MessageRepository:
    """创建消息仓储

    Args:
        session: 数据库会话

    Returns:
        MessageRepository 实例
    """
    return MessageRepository(session)


# ============== 健康检查 ==============


async def health_check() -> bool:
    """检查数据库连接健康状态

    Returns:
        数据库是否可用
    """
    try:
        async with session_scope() as session:
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        return False


# ============== 便捷函数 ==============


async def close_db():
    """关闭数据库连接"""
    global _async_engine, _sync_engine, _session_factory

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("async_db_closed")

    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
        logger.info("sync_db_closed")

    _session_factory = None
