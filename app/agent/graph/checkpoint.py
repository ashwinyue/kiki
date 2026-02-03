"""LangGraph Checkpoint 持久化

提供 AsyncPostgresSaver 的正确初始化和管理。
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.memory import MemorySaver

# 尝试导入 AsyncPostgresSaver（可选依赖）
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _postgres_available = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore
    _postgres_available = False

if TYPE_CHECKING:
    if AsyncPostgresSaver is None:
        # 类型检查时使用 Any 作为后备
        from typing import Any as AsyncPostgresSaver  # type: ignore

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

# 全局 checkpointer 单例
_postgres_checkpointer: "AsyncPostgresSaver | None" = None


async def get_postgres_checkpointer() -> "AsyncPostgresSaver | None":
    """获取 PostgreSQL checkpointer（单例）

    Returns:
        AsyncPostgresSaver 实例

    Raises:
        RuntimeError: PostgreSQL checkpointer 不可用时
    """
    global _postgres_checkpointer

    if _postgres_checkpointer is not None:
        return _postgres_checkpointer

    try:
        conn_string = _get_postgres_connection_string()
        _postgres_checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)

        await _postgres_checkpointer.setup()

        logger.info(
            "postgres_checkpointer_initialized",
            connection_string=_mask_connection_string(conn_string),
        )

        return _postgres_checkpointer

    except ImportError as e:
        logger.warning(
            "postgres_checkpointer_import_failed",
            error=str(e),
            hint="Install langgraph-checkpoint-postgres",
        )
        raise RuntimeError(
            "langgraph-checkpoint-postgres not installed. "
            "Run: uv add langgraph-checkpoint-postgres"
        ) from e
    except Exception as e:
        logger.error(
            "postgres_checkpointer_init_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(f"Failed to initialize PostgreSQL checkpointer: {e}") from e


async def get_checkpointer(use_postgres: bool = True) -> BaseCheckpointSaver:
    """获取 checkpointer 实例

    Args:
        use_postgres: 是否使用 PostgreSQL checkpointer。
                     如果为 True 但初始化失败，会自动降级到 MemorySaver

    Returns:
        BaseCheckpointSaver 实例（AsyncPostgresSaver 或 MemorySaver）
    """
    if use_postgres:
        try:
            return await get_postgres_checkpointer()
        except Exception as e:
            logger.warning(
                "postgres_checkpointer_fallback",
                error=str(e),
                fallback="memory_saver",
            )
            # 降级到 MemorySaver
            return MemorySaver()

    # 默认使用 MemorySaver
    logger.debug("using_memory_checkpointer")
    return MemorySaver()


@asynccontextmanager
async def checkpointer_context(
    use_postgres: bool = True,
) -> AsyncGenerator[BaseCheckpointSaver]:
    """Checkpointer 上下文管理器

    用于在应用生命周期中管理 checkpointer。

    Args:
        use_postgres: 是否使用 PostgreSQL checkpointer

    Yields:
        BaseCheckpointSaver 实例

    Examples:
        ```python
        async with checkpointer_context() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = await graph.ainvoke(...)
        ```
    """
    checkpointer = await get_checkpointer(use_postgres=use_postgres)
    try:
        yield checkpointer
    finally:
        # 注意：不要关闭 checkpointer，它是全局单例
        pass


async def close_postgres_checkpointer() -> None:
    """关闭 PostgreSQL checkpointer

    应用关闭时调用，释放资源。
    """
    global _postgres_checkpointer

    if _postgres_checkpointer is not None:
        try:
            # AsyncPostgresSaver 可能没有 close 方法
            # 它使用连接池，会自动管理连接
            _postgres_checkpointer = None
            logger.info("postgres_checkpointer_closed")
        except Exception as e:
            logger.warning(
                "postgres_checkpointer_close_error",
                error=str(e),
            )


def _get_postgres_connection_string() -> str:
    """获取 PostgreSQL 连接字符串

    从 settings.database_url 转换为 AsyncPostgresSaver 需要的格式。

    Returns:
        PostgreSQL 连接字符串

    Raises:
        ValueError: 数据库不是 PostgreSQL 时
    """
    db_url = settings.database_url

    # 如果已经是 postgresql:// 开头，直接返回
    if db_url.startswith("postgresql://") or db_url.startswith("postgresql+asyncpg://"):
        # 移除 asyncpg 驱动前缀，AsyncPostgresSaver 使用 psycopg
        if db_url.startswith("postgresql+asyncpg://"):
            return db_url.replace("postgresql+asyncpg://", "postgresql://")
        return db_url

    # 如果是 SQLite，抛出错误
    if db_url.startswith("sqlite"):
        raise ValueError(
            "PostgreSQL checkpointer requires PostgreSQL database, "
            "not SQLite. Use MemorySaver for SQLite."
        )

    raise ValueError(f"Unsupported database URL: {db_url}")


def _mask_connection_string(conn_string: str) -> str:
    """掩码连接字符串用于日志输出

    Args:
        conn_string: 原始连接字符串

    Returns:
        掩码后的连接字符串
    """
    try:
        # 移除密码部分
        if "://" in conn_string and "@" in conn_string:
            protocol, rest = conn_string.split("://", 1)
            if ":" in rest:
                user_part, host_part = rest.split("@", 1)
                user, _ = user_part.split(":", 1) if ":" in user_part else (user_part, "")
                return f"{protocol}://{user}:****@{host_part}"
        return conn_string.split("@")[1] if "@" in conn_string else conn_string
    except Exception:
        return "****"


# ============== 便捷函数 ==============


async def list_checkpoints(
    thread_id: str,
    limit: int = 10,
) -> list[Checkpoint]:
    """列出指定线程的检查点

    Args:
        thread_id: 线程 ID
        limit: 返回数量限制

    Returns:
        检查点列表
    """
    try:
        checkpointer = await get_postgres_checkpointer()

        # 获取配置
        config = {"configurable": {"thread_id": thread_id}}

        # 列出检查点（从最新的开始）
        checkpoints = []
        async for checkpoint in checkpointer.alist(config, limit=limit):
            checkpoints.append(checkpoint)

        return checkpoints

    except Exception as e:
        logger.error(
            "list_checkpoints_failed",
            thread_id=thread_id,
            error=str(e),
        )
        return []


async def get_checkpoint_count(thread_id: str) -> int:
    """获取指定线程的检查点数量

    Args:
        thread_id: 线程 ID

    Returns:
        检查点数量
    """
    try:
        checkpointer = await get_postgres_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}

        count = 0
        async for _ in checkpointer.alist(config):
            count += 1

        return count

    except Exception as e:
        logger.error(
            "get_checkpoint_count_failed",
            thread_id=thread_id,
            error=str(e),
        )
        return 0


__all__ = [
    # 主要函数
    "get_postgres_checkpointer",
    "get_checkpointer",
    "checkpointer_context",
    "close_postgres_checkpointer",
    # 查询函数
    "list_checkpoints",
    "get_checkpoint_count",
]
