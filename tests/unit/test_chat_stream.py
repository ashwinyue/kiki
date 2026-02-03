"""ChatStreamManager 单元测试

测试流式对话消息管理器的核心功能：
- 消息缓存
- 消息合并
- 数据库持久化
- 对话历史查询
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph.chat_stream import (
    ChatStreamManager,
    get_chat_stream_manager,
    reset_chat_stream_manager,
)


@pytest.fixture
async def mock_session():
    """模拟数据库会话"""
    session = AsyncMock(spec=AsyncSession)

    # 模拟 execute 返回值
    mock_result = AsyncMock()
    mock_result.first.return_value = None
    session.execute.return_value = mock_result
    session.commit = AsyncMock()

    return session


@pytest.fixture
async def chat_stream_manager(mock_session):
    """创建 ChatStreamManager 实例"""
    manager = ChatStreamManager(
        checkpoint_saver=True,
        db_url="postgresql://localhost:5432/test",
    )
    manager._session_factory = MagicMock()
    manager._session_factory.return_value.__aenter__.return_value = mock_session
    manager._session_factory.return_value.__aexit__ = AsyncMock()

    await manager.initialize()
    return manager


class TestChatStreamManager:
    """ChatStreamManager 测试类"""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_session):
        """测试初始化"""
        manager = ChatStreamManager(
            checkpoint_saver=True,
            db_url="postgresql://localhost:5432/test",
        )

        with patch("app.agent.graph.chat_stream.get_session_factory") as mock_factory:
            mock_factory.return_value = MagicMock()
            mock_factory.return_value.__aenter__.return_value = mock_session
            mock_factory.return_value.__aexit__ = AsyncMock()

            await manager.initialize()

            assert manager._session_factory is not None

    @pytest.mark.asyncio
    async def test_process_stream_message_cache(self, chat_stream_manager):
        """测试消息缓存（不触发持久化）"""
        thread_id = "test-thread-123"
        message = "Hello, world!"

        # 处理消息（不触发持久化）
        success = await chat_stream_manager.process_stream_message(
            thread_id=thread_id,
            message=message,
            finish_reason="",
        )

        assert success is True

        # 验证消息已缓存
        store_namespace = ("messages", thread_id)
        cursor = chat_stream_manager.store.get(store_namespace, "cursor")
        assert cursor is not None
        assert cursor.value["index"] == 0

        chunk = chat_stream_manager.store.get(store_namespace, "chunk_0")
        assert chunk is not None
        assert chunk.value == message

    @pytest.mark.asyncio
    async def test_process_stream_message_persist(self, chat_stream_manager, mock_session):
        """测试消息持久化（触发 finish_reason）"""
        thread_id = "test-thread-456"
        message = "Complete conversation"

        # 模拟数据库操作
        mock_result = MagicMock()
        mock_result.first.return_value = None  # 不存在现有记录
        mock_session.execute.return_value = mock_result

        # 处理消息（触发持久化）
        success = await chat_stream_manager.process_stream_message(
            thread_id=thread_id,
            message=message,
            finish_reason="stop",
        )

        assert success is True
        assert mock_session.execute.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_process_multiple_messages(self, chat_stream_manager):
        """测试多条消息处理"""
        thread_id = "test-thread-789"

        # 处理多条消息
        await chat_stream_manager.process_stream_message(
            thread_id=thread_id,
            message="Hello",
            finish_reason="",
        )
        await chat_stream_manager.process_stream_message(
            thread_id=thread_id,
            message=" World",
            finish_reason="",
        )
        await chat_stream_manager.process_stream_message(
            thread_id=thread_id,
            message="!",
            finish_reason="stop",
        )

        # 验证索引递增
        store_namespace = ("messages", thread_id)
        cursor = chat_stream_manager.store.get(store_namespace, "cursor")
        assert cursor.value["index"] == 2

    @pytest.mark.asyncio
    async def test_invalid_inputs(self, chat_stream_manager):
        """测试无效输入处理"""
        # 空 thread_id
        success = await chat_stream_manager.process_stream_message(
            thread_id="",
            message="test",
            finish_reason="",
        )
        assert success is False

        # 空消息
        success = await chat_stream_manager.process_stream_message(
            thread_id="thread-123",
            message="",
            finish_reason="",
        )
        assert success is False

    @pytest.mark.asyncio
    async def test_get_chat_history(self, chat_stream_manager, mock_session):
        """测试获取对话历史"""
        thread_id = "test-thread-history"

        # 模拟数据库返回
        expected_messages = ["Hello", "World"]
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: json.dumps(expected_messages)
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        mock_session.execute.return_value = mock_result

        # 获取历史
        messages = await chat_stream_manager.get_chat_history(thread_id)

        assert messages == expected_messages

    @pytest.mark.asyncio
    async def test_get_chat_history_not_found(self, chat_stream_manager, mock_session):
        """测试获取不存在的对话历史"""
        thread_id = "nonexistent-thread"

        # 模拟数据库返回空
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result

        # 获取历史
        messages = await chat_stream_manager.get_chat_history(thread_id)

        assert messages is None

    @pytest.mark.asyncio
    async def test_delete_chat_history(self, chat_stream_manager, mock_session):
        """测试删除对话历史"""
        thread_id = "test-thread-delete"

        # 模拟数据库操作
        mock_session.execute.return_value = MagicMock()

        # 删除历史
        success = await chat_stream_manager.delete_chat_history(thread_id)

        assert success is True
        assert mock_session.execute.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_list_recent_threads(self, chat_stream_manager, mock_session):
        """测试列出最近线程"""
        # 模拟数据库返回
        mock_rows = [
            ("thread-1", 5, "2025-02-03T10:00:00Z", "2025-02-03T09:00:00Z", "2025-02-03T10:00:00Z"),
            ("thread-2", 3, "2025-02-03T09:00:00Z", "2025-02-03T08:00:00Z", "2025-02-03T09:00:00Z"),
        ]
        mock_result = MagicMock()
        mock_result.all.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        # 列出线程
        threads = await chat_stream_manager.list_recent_threads(limit=10)

        assert len(threads) == 2
        assert threads[0]["thread_id"] == "thread-1"

    @pytest.mark.asyncio
    async def test_close(self, chat_stream_manager):
        """测试关闭管理器"""
        # 关闭管理器
        await chat_stream_manager.close()

        assert chat_stream_manager.store is None
        assert chat_stream_manager._session_factory is None


class TestGlobalInstance:
    """全局实例测试"""

    @pytest.mark.asyncio
    async def test_get_chat_stream_manager_singleton(self):
        """测试单例模式"""
        reset_chat_stream_manager()

        # 第一次获取
        manager1 = await get_chat_stream_manager()
        # 第二次获取
        manager2 = await get_chat_stream_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_reset_chat_stream_manager(self):
        """测试重置全局实例"""
        reset_chat_stream_manager()

        manager = await get_chat_stream_manager()
        assert manager is not None

        # 重置
        reset_chat_stream_manager()

        # 再次获取应该是新实例
        new_manager = await get_chat_stream_manager()
        assert new_manager is not manager

    @pytest.mark.asyncio
    async def test_checkpoint_saver_disabled(self):
        """测试禁用 checkpoint_saver"""
        reset_chat_stream_manager()

        # 模拟禁用配置
        with patch("app.agent.graph.chat_stream.settings") as mock_settings:
            mock_settings.chat_stream_checkpoint_saver = False

            manager = await get_chat_stream_manager()

            assert manager.checkpoint_saver is False

            # 禁用时不应该调用数据库
            success = await manager.process_stream_message(
                thread_id="test",
                message="test",
                finish_reason="stop",
            )
            # 禁用时不处理但也不报错
            assert success is False

        reset_chat_stream_manager()


@pytest.mark.integration
class TestChatStreamManagerIntegration:
    """集成测试（需要真实数据库）"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流"""
        # 注意：需要真实数据库连接
        # 这个测试可以在 CI 环境中运行
        pass

    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """测试并发流处理"""
        # 测试多个并发的流式会话
        pass
