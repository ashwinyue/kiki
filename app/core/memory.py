"""会话上下文存储模块

提供会话历史的多轮对话记忆功能，支持 Redis 和内存存储。
参考 WeKnora 的 ContextStorage 架构。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Optional
from collections.abc import AsyncIterator

from pydantic import BaseModel
from langchain_core.messages import BaseMessage

from app.core.redis import RedisCache, get_cache
from app.core.logging import get_logger


logger = get_logger(__name__)


# ============== 数据模型 ==============

class ChatMessage(BaseModel):
    """聊天消息模型"""

    role: str
    content: str
    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {}

    @classmethod
    def from_base_message(cls, message: BaseMessage) -> ChatMessage:
        """从 LangChain BaseMessage 转换

        Args:
            message: LangChain 消息

        Returns:
            ChatMessage 实例
        """
        return cls(
            role=message.type,
            content=message.content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
        )

    def to_base_message(self) -> BaseMessage:
        """转换为 LangChain BaseMessage

        Returns:
            BaseMessage 实例
        """
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )

        message_classes = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }

        msg_class = message_classes.get(self.role, HumanMessage)
        return msg_class(
            content=self.content,
            additional_kwargs=self.additional_kwargs,
            response_metadata=self.response_metadata,
        )


# ============== 存储接口 ==============

class ContextStorage(ABC):
    """会话上下文存储接口

    定义会话历史的存储操作，支持多种后端实现。
    """

    @abstractmethod
    async def save(self, session_id: str, messages: list[ChatMessage]) -> None:
        """保存会话消息

        Args:
            session_id: 会话 ID
            messages: 消息列表
        """
        pass

    @abstractmethod
    async def load(self, session_id: str) -> list[ChatMessage]:
        """加载会话消息

        Args:
            session_id: 会话 ID

        Returns:
            消息列表，不存在时返回空列表
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """删除会话消息

        Args:
            session_id: 会话 ID
        """
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """检查会话是否存在

        Args:
            session_id: 会话 ID

        Returns:
            是否存在
        """
        pass


# ============== Redis 存储 ==============

class RedisContextStorage(ContextStorage):
    """Redis 会话上下文存储

    使用 Redis 存储会话历史，支持自动过期。
    """

    def __init__(
        self,
        ttl: timedelta = timedelta(hours=24),
        key_prefix: str = "chat:context:",
    ) -> None:
        """初始化 Redis 存储

        Args:
            ttl: 过期时间，默认 24 小时
            key_prefix: 键前缀
        """
        self._cache = RedisCache(key_prefix=key_prefix)
        self._ttl = int(ttl.total_seconds())

    def _make_key(self, session_id: str) -> str:
        """生成 Redis 键"""
        return session_id

    async def save(self, session_id: str, messages: list[ChatMessage]) -> None:
        """保存会话消息到 Redis

        Args:
            session_id: 会话 ID
            messages: 消息列表
        """
        try:
            # 序列化为 JSON
            data = [msg.model_dump(mode="json") for msg in messages]
            json_str = json.dumps(data, ensure_ascii=False)

            # 保存到 Redis
            await self._cache.set(session_id, json_str, ttl=self._ttl)

            logger.debug(
                "redis_context_saved",
                session_id=session_id,
                message_count=len(messages),
                ttl=self._ttl,
            )

        except Exception as e:
            logger.error("redis_context_save_failed", session_id=session_id, error=str(e))
            raise

    async def load(self, session_id: str) -> list[ChatMessage]:
        """从 Redis 加载会话消息

        Args:
            session_id: 会话 ID

        Returns:
            消息列表
        """
        try:
            json_str = await self._cache.get(session_id)

            if json_str is None:
                logger.debug("redis_context_not_found", session_id=session_id)
                return []

            # 反序列化
            data = json.loads(json_str)
            messages = [ChatMessage(**item) for item in data]

            logger.debug(
                "redis_context_loaded",
                session_id=session_id,
                message_count=len(messages),
            )

            return messages

        except Exception as e:
            logger.error("redis_context_load_failed", session_id=session_id, error=str(e))
            return []

    async def delete(self, session_id: str) -> None:
        """从 Redis 删除会话消息

        Args:
            session_id: 会话 ID
        """
        try:
            await self._cache.delete(session_id)
            logger.debug("redis_context_deleted", session_id=session_id)

        except Exception as e:
            logger.error("redis_context_delete_failed", session_id=session_id, error=str(e))

    async def exists(self, session_id: str) -> bool:
        """检查会话是否存在

        Args:
            session_id: 会话 ID

        Returns:
            是否存在
        """
        try:
            count = await self._cache.exists(session_id)
            return count > 0

        except Exception as e:
            logger.error("redis_context_exists_failed", session_id=session_id, error=str(e))
            return False


# ============== 内存存储 ==============

class MemoryContextStorage(ContextStorage):
    """内存会话上下文存储

    使用进程内存存储，适用于开发/测试环境。
    """

    def __init__(self) -> None:
        """初始化内存存储"""
        self._storage: dict[str, list[ChatMessage]] = {}

    async def save(self, session_id: str, messages: list[ChatMessage]) -> None:
        """保存会话消息到内存

        Args:
            session_id: 会话 ID
            messages: 消息列表
        """
        # 深拷贝避免外部修改
        self._storage[session_id] = [
            ChatMessage(**msg.model_dump()) for msg in messages
        ]

        logger.debug(
            "memory_context_saved",
            session_id=session_id,
            message_count=len(messages),
        )

    async def load(self, session_id: str) -> list[ChatMessage]:
        """从内存加载会话消息

        Args:
            session_id: 会话 ID

        Returns:
            消息列表
        """
        messages = self._storage.get(session_id, [])

        # 返回副本
        return [ChatMessage(**msg.model_dump()) for msg in messages]

    async def delete(self, session_id: str) -> None:
        """从内存删除会话消息

        Args:
            session_id: 会话 ID
        """
        self._storage.pop(session_id, None)
        logger.debug("memory_context_deleted", session_id=session_id)

    async def exists(self, session_id: str) -> bool:
        """检查会话是否存在

        Args:
            session_id: 会话 ID

        Returns:
            是否存在
        """
        return session_id in self._storage

    async def list_sessions(self) -> list[str]:
        """列出所有会话 ID

        Returns:
            会话 ID 列表
        """
        return list(self._storage.keys())

    def clear_all(self) -> None:
        """清空所有会话"""
        self._storage.clear()
        logger.info("memory_context_cleared_all")


# ============== 上下文管理器 ==============

class ContextManager:
    """会话上下文管理器

    提供会话历史的增删改查功能，支持消息压缩和 Token 管理。
    """

    def __init__(
        self,
        storage: ContextStorage,
        max_messages: int = 100,
        max_tokens: int = 128_000,
    ) -> None:
        """初始化上下文管理器

        Args:
            storage: 存储后端
            max_messages: 最大消息数量（用于滑动窗口）
            max_tokens: 最大 Token 数量
        """
        self._storage = storage
        self._max_messages = max_messages
        self._max_tokens = max_tokens

    async def add_message(
        self,
        session_id: str,
        message: ChatMessage | BaseMessage,
    ) -> None:
        """添加消息到会话

        Args:
            session_id: 会话 ID
            message: 消息
        """
        # 转换消息格式
        if isinstance(message, BaseMessage):
            chat_msg = ChatMessage.from_base_message(message)
        else:
            chat_msg = message

        # 加载现有消息
        messages = await self._storage.load(session_id)
        messages.append(chat_msg)

        # 压缩处理
        compressed = await self._compress(messages)

        # 保存
        await self._storage.save(session_id, compressed)

        logger.info(
            "context_message_added",
            session_id=session_id,
            role=chat_msg.role,
            total_count=len(compressed),
        )

    async def get_context(
        self,
        session_id: str,
        as_base_messages: bool = False,
    ) -> list[ChatMessage] | list[BaseMessage]:
        """获取会话上下文

        Args:
            session_id: 会话 ID
            as_base_messages: 是否返回 LangChain BaseMessage

        Returns:
            消息列表
        """
        messages = await self._storage.load(session_id)

        if as_base_messages:
            return [msg.to_base_message() for msg in messages]

        return messages

    async def clear_context(self, session_id: str) -> None:
        """清空会话上下文

        Args:
            session_id: 会话 ID
        """
        await self._storage.delete(session_id)
        logger.info("context_cleared", session_id=session_id)

    async def get_stats(self, session_id: str) -> dict[str, Any]:
        """获取会话统计信息

        Args:
            session_id: 会话 ID

        Returns:
            统计信息
        """
        messages = await self._storage.load(session_id)

        # 统计角色分布
        role_count: dict[str, int] = {}
        for msg in messages:
            role_count[msg.role] = role_count.get(msg.role, 0) + 1

        # 估算 Token 数
        token_count = self._estimate_tokens(messages)

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "token_estimate": token_count,
            "role_distribution": role_count,
            "exists": await self._storage.exists(session_id),
        }

    async def _compress(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """压缩消息列表

        使用滑动窗口策略，保留最新的消息。

        Args:
            messages: 消息列表

        Returns:
            压缩后的消息列表
        """
        # 估算 Token
        token_count = self._estimate_tokens(messages)

        if token_count <= self._max_tokens and len(messages) <= self._max_messages:
            return messages

        # 滑动窗口压缩
        if len(messages) > self._max_messages:
            compressed = messages[-self._max_messages :]
            logger.info(
                "context_compressed",
                original_count=len(messages),
                compressed_count=len(compressed),
                reason="max_messages",
            )
            return compressed

        # Token 数量压缩
        result = []
        current_tokens = 0
        for msg in reversed(messages):
            msg_tokens = self._estimate_tokens([msg])
            if current_tokens + msg_tokens > self._max_tokens:
                break
            result.append(msg)
            current_tokens += msg_tokens

        compressed = list(reversed(result))
        logger.info(
            "context_compressed",
            original_count=len(messages),
            compressed_count=len(compressed),
            original_tokens=token_count,
            compressed_tokens=current_tokens,
            reason="max_tokens",
        )
        return compressed

    def _estimate_tokens(self, messages: list[ChatMessage]) -> int:
        """估算 Token 数量

        简单估算：中文字符 * 1.5 + 英文单词 * 1

        Args:
            messages: 消息列表

        Returns:
            估算的 Token 数
        """
        import re

        total = 0
        for msg in messages:
            content = msg.content
            # 中文字符
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
            # 英文单词
            english_words = len(re.findall(r"\b[a-zA-Z]+\b", content))
            total += int(chinese_chars * 1.5 + english_words)

        return total


# ============== 工厂函数 ==============

# 全局存储实例
_storage: ContextStorage | None = None
_manager: ContextManager | None = None


def get_context_storage(
    storage_type: str = "memory",
    ttl: timedelta = timedelta(hours=24),
) -> ContextStorage:
    """获取上下文存储实例

    Args:
        storage_type: 存储类型 (memory, redis)
        ttl: Redis 过期时间

    Returns:
        ContextStorage 实例
    """
    global _storage

    if _storage is None:
        if storage_type == "redis":
            _storage = RedisContextStorage(ttl=ttl)
        else:
            _storage = MemoryContextStorage()

        logger.info(
            "context_storage_initialized",
            storage_type=storage_type,
        )

    return _storage


def get_context_manager(
    storage_type: str = "memory",
    max_messages: int = 100,
    max_tokens: int = 128_000,
) -> ContextManager:
    """获取上下文管理器实例

    Args:
        storage_type: 存储类型 (memory, redis)
        max_messages: 最大消息数量
        max_tokens: 最大 Token 数量

    Returns:
        ContextManager 实例
    """
    global _manager

    if _manager is None:
        storage = get_context_storage(storage_type=storage_type)
        _manager = ContextManager(
            storage=storage,
            max_messages=max_messages,
            max_tokens=max_tokens,
        )

    return _manager
