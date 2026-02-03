"""数据库服务模块

提供数据库连接池、会话管理和事务处理。

采用单例模式管理连接池，确保资源高效利用。
参考外部项目的连接池单例设计模式。
"""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from threading import Lock
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

logger = get_logger(__name__)

settings = get_settings()

# 同步引擎（用于迁移）
_sync_engine = None

# 异步引擎
_async_engine: AsyncEngine | None = None

# 会话工厂
_session_factory = None

# 线程锁（确保单例线程安全）
_engine_lock = Lock()
_session_factory_lock = Lock()


# ============== 连接池单例管理器 ==============


class DatabaseConnectionPool:
    """数据库连接池单例管理器

    采用线程安全的单例模式，确保全局只有一个连接池实例。

    设计模式参考：
    - 外部项目的 MultiTenantVectorStore 单例模式
    - 懒初始化 + 线程安全

    使用示例:
        ```python
        pool = DatabaseConnectionPool()
        engine = pool.get_async_engine()
        ```
    """

    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        """线程安全的单例实现"""
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """懒初始化连接池"""
        if self._initialized:
            return

        self._async_engine: AsyncEngine | None = None
        self._sync_engine: Any = None
        self._session_factory: async_sessionmaker[Any] | None = None
        self._initialized = True

        logger.debug("database_pool_singleton_created")

    def get_async_engine(self) -> AsyncEngine:
        """获取异步数据库引擎（单例）

        Returns:
            异步引擎实例
        """
        if self._async_engine is None:
            with _engine_lock:
                if self._async_engine is None:
                    self._async_engine = create_async_engine(
                        settings.database_url,
                        echo=settings.database_echo,
                        pool_size=settings.database_pool_size,
                        pool_pre_ping=True,  # 连接前检查有效性
                        pool_recycle=3600,  # 1 小时回收连接
                    )
                    logger.info(
                        "async_db_engine_created",
                        pool_size=settings.database_pool_size,
                    )
        return self._async_engine

    def get_sync_engine(self):
        """获取同步数据库引擎（单例，用于迁移）

        Returns:
            同步引擎实例
        """
        if self._sync_engine is None:
            with _engine_lock:
                if self._sync_engine is None:
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

                    from sqlalchemy.engine import Engine

                    self._sync_engine: Engine = create_engine(db_url, **engine_args)
                    logger.info("sync_db_engine_created")
        return self._sync_engine

    def get_session_factory(self) -> async_sessionmaker[Any]:
        """获取会话工厂（单例）

        Returns:
            会话工厂实例
        """
        if self._session_factory is None:
            with _session_factory_lock:
                if self._session_factory is None:
                    engine = self.get_async_engine()
                    self._session_factory = async_sessionmaker(
                        bind=engine,
                        class_=AsyncSession,
                        expire_on_commit=False,
                        autocommit=False,
                        autoflush=False,
                    )
                    logger.debug("session_factory_created")
        return self._session_factory

    async def close(self) -> None:
        """关闭所有连接

        应用关闭时调用，释放资源。
        """
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            logger.info("async_db_closed")

        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            logger.info("sync_db_closed")

        self._session_factory = None
        self._initialized = False

        logger.info("database_pool_closed")


# 全局连接池单例实例
_db_pool = DatabaseConnectionPool()


def get_sync_engine():
    """获取同步数据库引擎

    兼容函数，内部使用单例管理器。

    Returns:
        同步引擎实例
    """
    return _db_pool.get_sync_engine()


def get_async_engine() -> AsyncEngine:
    """获取异步数据库引擎

    兼容函数，内部使用单例管理器。

    Returns:
        异步引擎实例
    """
    return _db_pool.get_async_engine()


def _get_session_factory():
    """获取会话工厂

    兼容函数，内部使用单例管理器。
    """
    return _db_pool.get_session_factory()


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


# ============== 仓储工厂方法 (已移除，避免循环导入) ==============

# 这些工厂方法已移到各自的 Repository 类中
# 直接使用: UserRepository(session) 替代 user_repository(session)


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
    """关闭数据库连接

    使用单例管理器统一管理连接关闭。
    """
    await _db_pool.close()
