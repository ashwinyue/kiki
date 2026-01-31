"""Memory 模块测试

测试短期和长期记忆的各种功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.memory.base import BaseLongTermMemory, BaseMemory
from app.agent.memory.short_term import ShortTermMemory, create_short_term_memory


class MockMemory(BaseMemory):
    """Mock Memory 实现用于测试"""

    def __init__(self) -> None:
        self._messages: list[AIMessage | HumanMessage] = []

    async def add_message(self, message: AIMessage | HumanMessage) -> None:
        self._messages.append(message)

    async def get_messages(self, limit: int | None = None) -> list[AIMessage | HumanMessage]:
        if limit:
            return self._messages[-limit:]
        return self._messages

    async def clear(self) -> None:
        self._messages.clear()

    async def count(self) -> int:
        return len(self._messages)


class MockLongTermMemory(BaseLongTermMemory):
    """Mock 长期记忆实现用于测试"""

    def __init__(self) -> None:
        self._memories: dict[str, dict] = {}
        self._counter = 0

    async def add_memory(self, content: str, metadata: dict | None = None) -> str:
        self._counter += 1
        memory_id = f"mem_{self._counter}"
        self._memories[memory_id] = {
            "content": content,
            "metadata": metadata or {},
        }
        return memory_id

    async def search_memories(
        self, query: str, k: int = 5, filter: dict | None = None
    ) -> list[dict]:
        # 简单的关键词匹配
        results = []
        for memory_id, memory in self._memories.items():
            if query.lower() in memory["content"].lower():
                results.append({
                    "id": memory_id,
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "score": 0.9,
                })
            if len(results) >= k:
                break
        return results

    async def delete_memory(self, memory_id: str) -> bool:
        return self._memories.pop(memory_id, None) is not None

    async def update_memory(
        self, memory_id: str, content: str | None = None, metadata: dict | None = None
    ) -> bool:
        if memory_id not in self._memories:
            return False
        if content:
            self._memories[memory_id]["content"] = content
        if metadata:
            self._memories[memory_id]["metadata"].update(metadata)
        return True


class TestBaseMemory:
    """BaseMemory 抽象基类测试"""

    @pytest.mark.asyncio
    async def test_mock_memory_implementation(self) -> None:
        """测试 Mock Memory 实现遵循接口"""
        memory = MockMemory()

        # 添加消息
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi there!")
        await memory.add_message(msg1)
        await memory.add_message(msg2)

        # 获取消息
        messages = await memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

        # 限制数量
        messages_limited = await memory.get_messages(limit=1)
        assert len(messages_limited) == 1
        assert messages_limited[0].content == "Hi there!"

        # 计数
        count = await memory.count()
        assert count == 2

        # 清除
        await memory.clear()
        assert await memory.count() == 0


class TestBaseLongTermMemory:
    """BaseLongTermMemory 抽象基类测试"""

    @pytest.mark.asyncio
    async def test_add_and_search_memory(self) -> None:
        """测试添加和搜索记忆"""
        memory = MockLongTermMemory()

        # 添加记忆
        memory_id = await memory.add_memory(
            content="Python 是一种编程语言",
            metadata={"category": "programming"},
        )
        assert memory_id == "mem_1"

        # 搜索记忆
        results = await memory.search_memories("Python")
        assert len(results) == 1
        assert results[0]["content"] == "Python 是一种编程语言"

    @pytest.mark.asyncio
    async def test_update_memory(self) -> None:
        """测试更新记忆"""
        memory = MockLongTermMemory()

        memory_id = await memory.add_memory("原始内容")

        # 更新内容
        success = await memory.update_memory(memory_id, content="更新后的内容")
        assert success is True

        results = await memory.search_memories("更新")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete_memory(self) -> None:
        """测试删除记忆"""
        memory = MockLongTermMemory()

        memory_id = await memory.add_memory("要删除的内容")

        # 删除
        success = await memory.delete_memory(memory_id)
        assert success is True

        # 再次删除应失败
        success = await memory.delete_memory(memory_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_search_with_limit(self) -> None:
        """测试带限制的搜索"""
        memory = MockLongTermMemory()

        # 添加多条记忆
        for i in range(10):
            await memory.add_memory(f"内容 {i}")

        results = await memory.search_memories("内容", k=3)
        assert len(results) == 3


class TestShortTermMemory:
    """ShortTermMemory 测试"""

    def test_init(self) -> None:
        """测试初始化"""
        memory = ShortTermMemory(session_id="test_session")
        assert memory.session_id == "test_session"
        assert memory._checkpointer is None

    def test_init_with_checkpointer(self) -> None:
        """测试使用 checkpointer 初始化"""
        mock_checkpointer = MagicMock()
        memory = ShortTermMemory(
            session_id="test_session",
            checkpointer=mock_checkpointer,
        )
        assert memory._checkpointer is mock_checkpointer

    @pytest.mark.asyncio
    async def test_get_checkpointer_when_provided(self) -> None:
        """测试获取提供的 checkpointer"""
        mock_checkpointer = MagicMock()
        memory = ShortTermMemory(
            session_id="test_session",
            checkpointer=mock_checkpointer,
        )

        result = await memory.get_checkpointer()
        assert result is mock_checkpointer

    @pytest.mark.asyncio
    async def test_add_message(self) -> None:
        """测试添加消息"""
        memory = ShortTermMemory(session_id="test_session")
        message = HumanMessage(content="Test message")

        # 应该不抛出异常
        await memory.add_message(message)

    @pytest.mark.asyncio
    async def test_get_messages_without_checkpointer(self) -> None:
        """测试无 checkpointer 时获取消息"""
        memory = ShortTermMemory(session_id="test_session")

        messages = await memory.get_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_clear_without_connection_pool(self) -> None:
        """测试无连接池时清除"""
        memory = ShortTermMemory(session_id="test_session")

        # 应该不抛出异常
        await memory.clear()

    @pytest.mark.asyncio
    async def test_close_without_connection_pool(self) -> None:
        """测试无连接池时关闭"""
        memory = ShortTermMemory(session_id="test_session")

        # 应该不抛出异常
        await memory.close()


class TestCreateShortTermMemory:
    """create_short_term_memory 函数测试"""

    def test_create_default(self) -> None:
        """测试创建默认短期记忆"""
        memory = create_short_term_memory(session_id="test_session")

        assert isinstance(memory, ShortTermMemory)
        assert memory.session_id == "test_session"

    def test_create_with_checkpointer(self) -> None:
        """测试使用 checkpointer 创建"""
        mock_checkpointer = MagicMock()
        memory = create_short_term_memory(
            session_id="test_session",
            checkpointer=mock_checkpointer,
        )

        assert memory._checkpointer is mock_checkpointer


@pytest.mark.parametrize("limit", [1, 5, 10, None])
async def test_memory_get_messages_with_various_limits(limit: int | None) -> None:
    """参数化测试各种获取消息限制"""
    memory = MockMemory()

    for i in range(10):
        await memory.add_message(HumanMessage(content=f"Message {i}"))

    messages = await memory.get_messages(limit=limit)
    if limit:
        assert len(messages) == limit
    else:
        assert len(messages) == 10
