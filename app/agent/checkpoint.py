"""检查点持久化

支持多种检查点存储后端：
- 内存（开发用）
- SQLite（小型部署）
- PostgreSQL（生产环境）
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.observability.logging import get_logger

# SQLite 检查点保存器（可选）
try:
    from langgraph.checkpoint.sqlite import AsyncSqliteSaver

    _sqlite_available = True
except ImportError:
    AsyncSqliteSaver = None  # type: ignore
    _sqlite_available = False

# PostgreSQL 检查点保存器（可选）
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _postgres_available = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore
    _postgres_available = False


logger = get_logger(__name__)


class CheckpointManager:
    """检查点管理器

    统一接口，支持多种存储后端。
    """

    def __init__(self, saver: BaseCheckpointSaver) -> None:
        self._saver = saver

    @classmethod
    def from_memory(cls) -> "CheckpointManager":
        """创建内存检查点管理器（开发用）"""
        return cls(MemorySaver())

    @classmethod
    def from_sqlite(cls, path: str) -> "CheckpointManager":
        """创建 SQLite 检查点管理器

        Args:
            path: SQLite 数据库路径

        Returns:
            CheckpointManager 实例

        Raises:
            ImportError: 如果 SQLite 检查点不可用
        """
        if not _sqlite_available:
            raise ImportError(
                "SQLite checkpoint saver is not available. Install langgraph-checkpoint-sqlite"
            )
        saver = AsyncSqliteSaver.from_conn_string(path)
        return cls(saver)

    @classmethod
    def from_postgres(cls, connection_string: str) -> "CheckpointManager":
        """创建 PostgreSQL 检查点管理器

        Args:
            connection_string: PostgreSQL 连接字符串

        Returns:
            CheckpointManager 实例

        Raises:
            ImportError: 如果 PostgreSQL 检查点不可用
        """
        if not _postgres_available:
            raise ImportError(
                "PostgreSQL checkpoint saver is not available. Install langgraph-checkpoint-postgres"
            )
        saver = AsyncPostgresSaver.from_conn_string(connection_string)
        return cls(saver)

    @property
    def saver(self) -> BaseCheckpointSaver:
        """获取底层 saver"""
        return self._saver

    async def close(self) -> None:
        """关闭连接"""
        if hasattr(self._saver, "close"):
            await self._saver.close()


def create_checkpointer(
    backend: str = "memory",
    connection_string: str | None = None,
) -> BaseCheckpointSaver:
    """创建检查点保存器（便捷函数）

    Args:
        backend: 存储后端 (memory/sqlite/postgres)
        connection_string: 连接字符串（SQLite/Postgres 需要）

    Returns:
        BaseCheckpointSaver 实例

    Examples:
        ```python
        # 内存检查点
        checkpointer = create_checkpointer("memory")

        # SQLite 检查点
        checkpointer = create_checkpointer("sqlite", "checkpoints.db")

        # PostgreSQL 检查点
        checkpointer = create_checkpointer("postgres", "postgresql://...")
        ```
    """
    if backend == "memory":
        manager = CheckpointManager.from_memory()
    elif backend == "sqlite":
        if not connection_string:
            raise ValueError("SQLite backend requires connection_string")
        manager = CheckpointManager.from_sqlite(connection_string)
    elif backend == "postgres":
        if not connection_string:
            raise ValueError("Postgres backend requires connection_string")
        manager = CheckpointManager.from_postgres(connection_string)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return manager.saver
