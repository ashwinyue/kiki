"""检查点持久化测试

测试 CheckpointManager 和 create_checkpointer 函数的各种功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.checkpoint import (
    CheckpointManager,
    create_checkpointer,
)


class TestCheckpointManager:
    """CheckpointManager 测试"""

    def test_from_memory(self) -> None:
        """测试创建内存检查点管理器"""
        manager = CheckpointManager.from_memory()

        assert manager._saver is not None
        assert hasattr(manager._saver, "put")
        assert hasattr(manager._saver, "get")

    def test_saver_property(self) -> None:
        """测试获取 saver 属性"""
        manager = CheckpointManager.from_memory()
        saver = manager.saver

        assert saver is not None
        assert saver is manager._saver

    @pytest.mark.asyncio
    async def test_close_memory_saver(self) -> None:
        """测试关闭内存 saver"""
        manager = CheckpointManager.from_memory()

        # MemorySaver 没有 close 方法，应该不报错
        await manager.close()

    @pytest.mark.asyncio
    async def test_close_with_close_method(self) -> None:
        """测试关闭带 close 方法的 saver"""
        mock_saver = MagicMock()
        mock_close = AsyncMock()
        mock_saver.close = mock_close

        manager = CheckpointManager(mock_saver)
        await manager.close()

        mock_close.assert_called_once()


class TestCreateCheckpointer:
    """create_checkpointer 函数测试"""

    def test_create_memory_checkpointer(self) -> None:
        """测试创建内存检查点"""
        checkpointer = create_checkpointer("memory")

        assert checkpointer is not None
        assert hasattr(checkpointer, "put")
        assert hasattr(checkpointer, "get")

    def test_create_sqlite_checkpointer_without_connection_string(self) -> None:
        """测试 SQLite 检查点缺少连接字符串"""
        with pytest.raises(ValueError, match="SQLite backend requires connection_string"):
            create_checkpointer("sqlite")

    def test_create_postgres_checkpointer_without_connection_string(self) -> None:
        """测试 PostgreSQL 检查点缺少连接字符串"""
        with pytest.raises(ValueError, match="Postgres backend requires connection_string"):
            create_checkpointer("postgres")

    def test_create_unknown_backend(self) -> None:
        """测试创建未知后端"""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_checkpointer("unknown_backend")

    @patch("app.agent.checkpoint._sqlite_available", True)
    @patch("app.agent.checkpoint.AsyncSqliteSaver")
    def test_create_sqlite_checkpointer_success(self, mock_saver_class) -> None:
        """测试成功创建 SQLite 检查点"""
        mock_saver = MagicMock()
        mock_saver_instance = MagicMock()
        mock_saver_class.from_conn_string.return_value = mock_saver_instance

        checkpointer = create_checkpointer("sqlite", "test.db")

        mock_saver_class.from_conn_string.assert_called_once_with("test.db")
        assert checkpointer is not None

    @patch("app.agent.checkpoint._sqlite_available", False)
    def test_create_sqlite_when_not_available(self) -> None:
        """测试 SQLite 不可用时抛出 ImportError"""
        with pytest.raises(ImportError, match="SQLite checkpoint saver is not available"):
            CheckpointManager.from_sqlite("test.db")

    @patch("app.agent.checkpoint._postgres_available", True)
    @patch("app.agent.checkpoint.AsyncPostgresSaver")
    def test_create_postgres_checkpointer_success(self, mock_saver_class) -> None:
        """测试成功创建 PostgreSQL 检查点"""
        mock_saver_instance = MagicMock()
        mock_saver_class.from_conn_string.return_value = mock_saver_instance

        checkpointer = create_checkpointer(
            "postgres", "postgresql://localhost/test"
        )

        mock_saver_class.from_conn_string.assert_called_once_with(
            "postgresql://localhost/test"
        )
        assert checkpointer is not None

    @patch("app.agent.checkpoint._postgres_available", False)
    def test_create_postgres_when_not_available(self) -> None:
        """测试 PostgreSQL 不可用时抛出 ImportError"""
        with pytest.raises(ImportError, match="PostgreSQL checkpoint saver is not available"):
            CheckpointManager.from_postgres("postgresql://localhost/test")


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_backend_parameter_validation(backend) -> None:
    """参数化测试各种后端"""
    if backend == "memory":
        checkpointer = create_checkpointer(backend)
        assert checkpointer is not None
    else:
        # 其他后端需要连接字符串
        with pytest.raises(ValueError):
            create_checkpointer(backend)
