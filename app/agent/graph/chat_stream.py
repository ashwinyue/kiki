"""流式对话消息持久化管理

参考 DeerFlow 的 ChatStreamManager 设计，实现：
- InMemoryStore 临时缓存流式消息
- PostgreSQL 持久化完整对话
- 按消息块索引存储，自动合并
- finish_reason 触发持久化

核心功能：
1. process_stream_message() - 处理流式消息
2. _persist_to_postgresql() - 持久化到数据库
3. get_chat_history() - 获取对话历史

使用示例：
    ```python
    manager = ChatStreamManager(checkpoint_saver=True)
    await manager.initialize()

    # 处理流式消息
    await manager.process_stream_message(
        thread_id="thread-123",
        message="Hello",
        finish_reason="stop"  # 触发持久化
    )

    # 获取历史
    history = await manager.get_chat_history("thread-123")
    ```
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.store.memory import InMemoryStore
from sqlalchemy import text

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


class ChatStreamManager:
    """流式对话消息管理器

    双层存储架构：
    1. InMemoryStore - 临时缓存流式消息块
    2. PostgreSQL - 持久化完整对话（在 finish_reason 触发时）

    参考：deer-flow/src/graph/checkpoint.py
    """

    def __init__(
        self,
        checkpoint_saver: bool = False,
        db_url: str | None = None,
    ) -> None:
        """初始化 ChatStreamManager

        Args:
            checkpoint_saver: 是否启用持久化
            db_url: 数据库连接字符串（可选，默认使用 settings.database_url）
        """
        self.checkpoint_saver = checkpoint_saver
        self.db_url = db_url or settings.database_url

        # 内存存储 - 临时缓存流式消息
        self.store = InMemoryStore()

        # 数据库会话（延迟初始化）
        self._session_factory = None

        logger.debug(
            "chat_stream_manager_created",
            checkpoint_saver=checkpoint_saver,
        )

    async def initialize(self) -> None:
        """初始化管理器

        - 确保数据库表存在
        - 建立数据库连接
        """
        if not self.checkpoint_saver:
            logger.info("checkpoint_saver_disabled")
            return

        try:
            from app.infra.database import get_session_factory

            self._session_factory = get_session_factory()

            # 创建表（如果不存在）
            await self._create_chat_streams_table()

            logger.info("chat_stream_manager_initialized")

        except Exception as e:
            logger.error(
                "chat_stream_manager_init_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _create_chat_streams_table(self) -> None:
        """创建 chat_streams 表（如果不存在）"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chat_streams (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            thread_id VARCHAR(255) NOT NULL UNIQUE,
            messages JSONB NOT NULL,
            ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- 索引
        CREATE INDEX IF NOT EXISTS idx_chat_streams_thread_id ON chat_streams(thread_id);
        CREATE INDEX IF NOT EXISTS idx_chat_streams_ts ON chat_streams(ts);
        CREATE INDEX IF NOT EXISTS idx_chat_streams_created_at ON chat_streams(created_at);

        -- 触发器：自动更新 updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        DROP TRIGGER IF EXISTS update_chat_streams_updated_at ON chat_streams;
        CREATE TRIGGER update_chat_streams_updated_at
            BEFORE UPDATE ON chat_streams
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """

        async with self._session_factory() as session:
            await session.execute(text(create_table_sql))
            await session.commit()

        logger.info("chat_streams_table_created")

    async def process_stream_message(
        self,
        thread_id: str,
        message: str,
        finish_reason: str = "",
    ) -> bool:
        """处理流式消息

        Args:
            thread_id: 会话线程 ID
            message: 消息内容（可以是完整消息或消息块）
            finish_reason: 完成原因 ("stop", "interrupt", 或其他)

        Returns:
            是否处理成功
        """
        if not thread_id or not isinstance(thread_id, str):
            logger.warning("invalid_thread_id", thread_id=thread_id)
            return False

        if not message:
            logger.warning("empty_message", thread_id=thread_id)
            return False

        try:
            # 创建命名空间
            store_namespace: tuple[str, str] = ("messages", thread_id)

            # 获取/初始化游标
            cursor = self.store.get(store_namespace, "cursor")
            current_index = 0

            if cursor is not None:
                current_index = int(cursor.value.get("index", 0)) + 1

            # 更新游标
            self.store.put(store_namespace, "cursor", {"index": current_index})

            # 存储消息块
            chunk_key = f"chunk_{current_index}"
            self.store.put(store_namespace, chunk_key, message)

            logger.debug(
                "stream_message_cached",
                thread_id=thread_id,
                chunk_index=current_index,
                message_length=len(message),
            )

            # 检查是否需要持久化
            if finish_reason in ("stop", "interrupt"):
                return await self._persist_complete_conversation(
                    thread_id,
                    store_namespace,
                    current_index,
                )

            return True

        except Exception as e:
            logger.error(
                "process_stream_message_failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def _persist_complete_conversation(
        self,
        thread_id: str,
        store_namespace: tuple[str, str],
        final_index: int,
    ) -> bool:
        """持久化完整对话到数据库

        Args:
            thread_id: 会话线程 ID
            store_namespace: 内存存储命名空间
            final_index: 最后一个消息块的索引

        Returns:
            是否持久化成功
        """
        if not self.checkpoint_saver:
            logger.debug("checkpoint_saver_disabled_skip_persist")
            return False

        try:
            # 从内存存储获取所有消息块
            memories = self.store.search(store_namespace, limit=final_index + 2)

            # 提取消息内容（过滤掉游标元数据）
            messages: list[str] = []
            for item in memories:
                value = item.dict().get("value", "")
                # 跳过游标元数据，只包含实际消息块
                if value and not isinstance(value, dict):
                    messages.append(str(value))

            if not messages:
                logger.warning("no_messages_to_persist", thread_id=thread_id)
                return False

            # 持久化到数据库
            return await self._persist_to_postgresql(thread_id, messages)

        except Exception as e:
            logger.error(
                "persist_conversation_failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def _persist_to_postgresql(
        self,
        thread_id: str,
        messages: list[str],
    ) -> bool:
        """持久化到 PostgreSQL

        Args:
            thread_id: 会话线程 ID
            messages: 消息列表

        Returns:
            是否持久化成功
        """
        if not self._session_factory:
            logger.warning("session_factory_not_initialized")
            return False

        try:
            async with self._session_factory() as session:
                # 检查是否已存在
                check_sql = text(
                    "SELECT id FROM chat_streams WHERE thread_id = :thread_id"
                )
                result = await session.execute(
                    check_sql, {"thread_id": thread_id}
                )
                existing = result.first()

                messages_json = json.dumps(messages, ensure_ascii=False)
                current_timestamp = datetime.now(UTC)

                if existing:
                    # 更新现有记录
                    update_sql = text("""
                        UPDATE chat_streams
                        SET messages = :messages,
                            updated_at = :updated_at
                        WHERE thread_id = :thread_id
                    """)
                    await session.execute(
                        update_sql,
                        {
                            "thread_id": thread_id,
                            "messages": messages_json,
                            "updated_at": current_timestamp,
                        },
                    )
                    await session.commit()

                    logger.info(
                        "chat_stream_updated",
                        thread_id=thread_id,
                        message_count=len(messages),
                    )
                else:
                    # 创建新记录
                    insert_sql = text("""
                        INSERT INTO chat_streams (id, thread_id, messages, ts)
                        VALUES (:id, :thread_id, :messages, :ts)
                    """)
                    await session.execute(
                        insert_sql,
                        {
                            "id": uuid.uuid4(),
                            "thread_id": thread_id,
                            "messages": messages_json,
                            "ts": current_timestamp,
                        },
                    )
                    await session.commit()

                    logger.info(
                        "chat_stream_created",
                        thread_id=thread_id,
                        message_count=len(messages),
                    )

                return True

        except Exception as e:
            logger.error(
                "persist_to_postgresql_failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def get_chat_history(
        self,
        thread_id: str,
    ) -> list[str] | None:
        """获取对话历史

        Args:
            thread_id: 会话线程 ID

        Returns:
            消息列表，不存在返回 None
        """
        if not self._session_factory:
            logger.warning("session_factory_not_initialized")
            return None

        try:
            async with self._session_factory() as session:
                select_sql = text("""
                    SELECT messages
                    FROM chat_streams
                    WHERE thread_id = :thread_id
                    ORDER BY ts DESC
                    LIMIT 1
                """)
                result = await session.execute(
                    select_sql, {"thread_id": thread_id}
                )
                row = result.first()

                if row:
                    messages = json.loads(row[0])
                    logger.debug(
                        "chat_history_retrieved",
                        thread_id=thread_id,
                        message_count=len(messages),
                    )
                    return messages

                return None

        except Exception as e:
            logger.error(
                "get_chat_history_failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    async def delete_chat_history(
        self,
        thread_id: str,
    ) -> bool:
        """删除对话历史

        Args:
            thread_id: 会话线程 ID

        Returns:
            是否删除成功
        """
        if not self._session_factory:
            logger.warning("session_factory_not_initialized")
            return False

        try:
            async with self._session_factory() as session:
                delete_sql = text(
                    "DELETE FROM chat_streams WHERE thread_id = :thread_id"
                )
                await session.execute(delete_sql, {"thread_id": thread_id})
                await session.commit()

                # 同时清理内存存储
                store_namespace = ("messages", thread_id)
                # 清空命名空间中的所有项
                for i in range(1000):  # 限制循环次数
                    try:
                        self.store.delete(store_namespace, f"chunk_{i}")
                    except KeyError:
                        break
                try:
                    self.store.delete(store_namespace, "cursor")
                except KeyError:
                    pass

                logger.info("chat_history_deleted", thread_id=thread_id)
                return True

        except Exception as e:
            logger.error(
                "delete_chat_history_failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def list_recent_threads(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """列出最近的对话线程

        Args:
            limit: 返回数量限制

        Returns:
            线程信息列表
        """
        if not self._session_factory:
            logger.warning("session_factory_not_initialized")
            return []

        try:
            async with self._session_factory() as session:
                select_sql = text("""
                    SELECT thread_id, COUNT(messages) as message_count,
                           ts, created_at, updated_at
                    FROM chat_streams
                    ORDER BY ts DESC
                    LIMIT :limit
                """)
                result = await session.execute(select_sql, {"limit": limit})
                rows = result.all()

                threads = [
                    {
                        "thread_id": row[0],
                        "message_count": len(json.loads(row[1])) if row[1] else 0,
                        "ts": row[2].isoformat() if row[2] else None,
                        "created_at": row[3].isoformat() if row[3] else None,
                        "updated_at": row[4].isoformat() if row[4] else None,
                    }
                    for row in rows
                ]

                logger.debug(
                    "recent_threads_listed",
                    count=len(threads),
                )
                return threads

        except Exception as e:
            logger.error(
                "list_recent_threads_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    async def close(self) -> None:
        """关闭管理器，释放资源"""
        self.store = None
        self._session_factory = None
        logger.info("chat_stream_manager_closed")


# ============== 全局实例 ==============

_chat_stream_manager: ChatStreamManager | None = None


async def get_chat_stream_manager() -> ChatStreamManager:
    """获取 ChatStreamManager 实例（单例）

    Returns:
        ChatStreamManager 实例
    """
    global _chat_stream_manager

    if _chat_stream_manager is None:
        checkpoint_saver = getattr(settings, "chat_stream_checkpoint_saver", False)
        _chat_stream_manager = ChatStreamManager(
            checkpoint_saver=checkpoint_saver,
        )
        await _chat_stream_manager.initialize()

    return _chat_stream_manager


def reset_chat_stream_manager() -> None:
    """重置全局实例（用于测试）"""
    global _chat_stream_manager
    _chat_stream_manager = None


__all__ = [
    "ChatStreamManager",
    "get_chat_stream_manager",
    "reset_chat_stream_manager",
]
