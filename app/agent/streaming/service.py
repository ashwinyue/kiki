"""流继续服务

提供活跃流式会话的管理和客户端重连支持。
对齐 WeKnora 的 sessions/continue-stream/{id} API 实现。

核心功能：
- 管理活跃的流式会话
- 支持客户端重连到正在进行的流
- 缓存流式事件供重连获取
- 超时处理和清理
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.observability.logging import get_logger

logger = get_logger(__name__)

# Redis 键前缀
_STREAM_BUFFER_PREFIX = "kiki:stream:buffer:"
_STREAM_META_PREFIX = "kiki:stream:meta:"

# 默认配置
_DEFAULT_BUFFER_SIZE = 1000  # 缓存最近 1000 个事件
_DEFAULT_EVENT_TTL = 300  # 事件缓存 5 分钟
_DEFAULT_STREAM_TIMEOUT = 600  # 流超时 10 分钟


class StreamEvent(BaseModel):
    """流事件模型

    Attributes:
        event_type: 事件类型 (token, tool_start, tool_end, error, metadata, done)
        content: 事件内容
        metadata: 额外元数据
        timestamp: 事件时间戳
    """

    event_type: str = Field(..., description="事件类型")
    content: str = Field("", description="事件内容")
    metadata: dict[str, Any] | None = Field(None, description="额外元数据")
    timestamp: float = Field(
        default_factory=lambda: datetime.now(UTC).timestamp(),
        description="事件时间戳",
    )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.event_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        """转换为 SSE 格式"""
        data = json.dumps(self.to_dict(), ensure_ascii=False)
        return f"event: {self.event_type}\ndata: {data}\n\n"


class StreamMetadata(BaseModel):
    """流元数据

    Attributes:
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        started_at: 开始时间
        updated_at: 更新时间
        is_active: 是否活跃
        event_count: 事件计数
    """

    session_id: str = Field(..., description="会话 ID")
    user_id: int | None = Field(None, description="用户 ID")
    tenant_id: int | None = Field(None, description="租户 ID")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="开始时间",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="更新时间",
    )
    is_active: bool = Field(True, description="是否活跃")
    event_count: int = Field(0, description="事件计数")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


class StreamContinuationService:
    """流继续服务

    管理活跃的流式会话，支持客户端重连。

    使用 Redis 存储流事件和元数据，支持分布式部署。

    Examples:
        ```python
        service = StreamContinuationService()

        # 注册新流
        await service.register_stream("session-123", user_id=1)

        # 添加事件
        await service.add_event("session-123", StreamEvent(
            event_type="token",
            content="Hello",
        ))

        # 获取事件迭代器（用于重连）
        async for event in service.get_events("session-123", since=0):
            yield event.to_sse()

        # 标记完成
        await service.complete_stream("session-123")
        ```
    """

    def __init__(
        self,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        event_ttl: int = _DEFAULT_EVENT_TTL,
        stream_timeout: int = _DEFAULT_STREAM_TIMEOUT,
    ) -> None:
        """初始化流继续服务

        Args:
            buffer_size: 内存缓冲区大小
            event_ttl: 事件过期时间（秒）
            stream_timeout: 流超时时间（秒）
        """
        from app.infra.redis import get_cache

        self._buffer_size = buffer_size
        self._event_ttl = event_ttl
        self._stream_timeout = stream_timeout
        self._buffer_cache = get_cache(key_prefix=_STREAM_BUFFER_PREFIX)
        self._meta_cache = get_cache(key_prefix=_STREAM_META_PREFIX)

        # 内存中的活跃流缓冲区（用于快速访问）
        self._active_streams: dict[str, deque[StreamEvent]] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}

    def _make_buffer_key(self, session_id: str) -> str:
        """生成缓冲区键"""
        return f"buffer:{session_id}"

    def _make_meta_key(self, session_id: str) -> str:
        """生成元数据键"""
        return f"meta:{session_id}"

    async def register_stream(
        self,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> bool:
        """注册新的流式会话

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            是否注册成功
        """
        try:
            # 创建元数据
            metadata = StreamMetadata(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                is_active=True,
                event_count=0,
            )

            # 保存到 Redis
            meta_key = self._make_meta_key(session_id)
            success = await self._meta_cache.set(
                meta_key,
                metadata.model_dump_json(),
                ttl=self._stream_timeout,
            )

            if success:
                # 初始化内存缓冲区
                self._active_streams[session_id] = deque(maxlen=self._buffer_size)
                logger.info(
                    "stream_registered",
                    session_id=session_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )

            return success

        except Exception as e:
            logger.error(
                "register_stream_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def is_stream_active(self, session_id: str) -> bool:
        """检查流是否活跃

        Args:
            session_id: 会话 ID

        Returns:
            True 表示流活跃
        """
        try:
            meta_key = self._make_meta_key(session_id)
            value = await self._meta_cache.get(meta_key)

            if value is None:
                return False

            data = json.loads(value)
            return data.get("is_active", False)

        except Exception as e:
            logger.error(
                "check_stream_active_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def add_event(
        self,
        session_id: str,
        event: StreamEvent,
    ) -> bool:
        """添加事件到流

        Args:
            session_id: 会话 ID
            event: 流事件

        Returns:
            是否添加成功
        """
        try:
            # 更新内存缓冲区
            if session_id in self._active_streams:
                self._active_streams[session_id].append(event)

            # 更新 Redis 缓冲区
            buffer_key = self._make_buffer_key(session_id)
            event_json = event.model_dump_json()

            # 使用 Redis List 存储事件（最多保留 buffer_size 个）
            from app.infra.redis import get_redis

            client = await get_redis()
            await client.lpush(buffer_key, event_json)
            await client.ltrim(buffer_key, 0, self._buffer_size - 1)
            await client.expire(buffer_key, self._event_ttl)

            # 更新元数据
            await self._update_metadata(session_id)

            return True

        except Exception as e:
            logger.error(
                "add_event_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def get_events(
        self,
        session_id: str,
        since: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """获取流事件

        Args:
            session_id: 会话 ID
            since: 从第几个事件开始（0 表示从头开始）
            limit: 最多返回多少个事件

        Yields:
            StreamEvent 实例
        """
        try:
            from app.infra.redis import get_redis

            client = await get_redis()
            buffer_key = self._make_buffer_key(session_id)

            # 从 Redis 获取事件（注意：lpush 后需要 lrange 获取）
            events_json = await client.lrange(buffer_key, 0, -1)

            # 反转列表（因为 lpush 导致顺序相反）
            events_json.reverse()

            # 过滤并返回
            count = 0
            for i, event_json in enumerate(events_json):
                if i < since:
                    continue

                if limit is not None and count >= limit:
                    break

                data = json.loads(event_json)
                yield StreamEvent(**data)
                count += 1

            # 检查内存缓冲区中是否有新事件
            if session_id in self._active_streams:
                memory_events = list(self._active_streams[session_id])
                for i, event in enumerate(memory_events):
                    event_index = len(events_json) + i
                    if event_index < since:
                        continue

                    if limit is not None and count >= limit:
                        break

                    yield event
                    count += 1

        except Exception as e:
            logger.error(
                "get_events_failed",
                session_id=session_id,
                error=str(e),
            )

    async def get_latest_events(
        self,
        session_id: str,
        count: int = 100,
    ) -> list[StreamEvent]:
        """获取最新的 N 个事件

        Args:
            session_id: 会话 ID
            count: 获取数量

        Returns:
            事件列表
        """
        events = []
        async for event in self.get_events(session_id, limit=count):
            events.append(event)
        return events

    async def get_metadata(self, session_id: str) -> StreamMetadata | None:
        """获取流元数据

        Args:
            session_id: 会话 ID

        Returns:
            流元数据，不存在返回 None
        """
        try:
            meta_key = self._make_meta_key(session_id)
            value = await self._meta_cache.get(meta_key)

            if value is None:
                return None

            data = json.loads(value)
            return StreamMetadata(**data)

        except Exception as e:
            logger.error(
                "get_metadata_failed",
                session_id=session_id,
                error=str(e),
            )
            return None

    async def complete_stream(
        self,
        session_id: str,
        final_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """标记流完成

        Args:
            session_id: 会话 ID
            final_metadata: 最终元数据

        Returns:
            是否标记成功
        """
        try:
            # 更新元数据
            metadata = await self.get_metadata(session_id)
            if metadata:
                metadata.is_active = False
                metadata.updated_at = datetime.now(UTC)

                meta_key = self._make_meta_key(session_id)
                await self._meta_cache.set(
                    meta_key,
                    metadata.model_dump_json(),
                    ttl=self._event_ttl,
                )

            # 添加完成事件
            done_event = StreamEvent(
                event_type="done",
                content="",
                metadata=final_metadata or {},
            )
            await self.add_event(session_id, done_event)

            # 清理内存缓冲区（延迟清理，允许重连）
            if session_id in self._stream_tasks:
                self._stream_tasks[session_id].cancel()

            async def cleanup_task() -> None:
                await asyncio.sleep(self._event_ttl)
                if session_id in self._active_streams:
                    del self._active_streams[session_id]
                if session_id in self._stream_tasks:
                    del self._stream_tasks[session_id]

            self._stream_tasks[session_id] = asyncio.create_task(cleanup_task())

            logger.info("stream_completed", session_id=session_id)
            return True

        except Exception as e:
            logger.error(
                "complete_stream_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def abort_stream(
        self,
        session_id: str,
        reason: str = "Stream aborted",
    ) -> bool:
        """中止流

        Args:
            session_id: 会话 ID
            reason: 中止原因

        Returns:
            是否中止成功
        """
        try:
            # 添加错误事件
            error_event = StreamEvent(
                event_type="error",
                content=reason,
                metadata={"aborted": True},
            )
            await self.add_event(session_id, error_event)

            # 标记完成
            return await self.complete_stream(
                session_id,
                final_metadata={"error": reason, "aborted": True},
            )

        except Exception as e:
            logger.error(
                "abort_stream_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def cleanup_stream(self, session_id: str) -> bool:
        """清理流数据

        Args:
            session_id: 会话 ID

        Returns:
            是否清理成功
        """
        try:
            from app.infra.redis import get_redis

            client = await get_redis()

            # 删除缓冲区和元数据
            buffer_key = self._make_buffer_key(session_id)
            meta_key = self._make_meta_key(session_id)

            await client.delete(buffer_key, meta_key)

            # 清理内存
            if session_id in self._active_streams:
                del self._active_streams[session_id]
            if session_id in self._stream_tasks:
                self._stream_tasks[session_id].cancel()
                del self._stream_tasks[session_id]

            logger.info("stream_cleaned", session_id=session_id)
            return True

        except Exception as e:
            logger.error(
                "cleanup_stream_failed",
                session_id=session_id,
                error=str(e),
            )
            return False

    async def _update_metadata(self, session_id: str) -> None:
        """更新元数据（增加事件计数）"""
        try:
            metadata = await self.get_metadata(session_id)
            if metadata:
                metadata.event_count += 1
                metadata.updated_at = datetime.now(UTC)

                meta_key = self._make_meta_key(session_id)
                await self._meta_cache.set(
                    meta_key,
                    metadata.model_dump_json(),
                    ttl=self._stream_timeout,
                )
        except Exception as e:
            logger.error(
                "update_metadata_failed",
                session_id=session_id,
                error=str(e),
            )


# 全局实例
_stream_continuation_service: StreamContinuationService | None = None


def get_stream_continuation_service() -> StreamContinuationService:
    """获取流继续服务实例（单例）

    Returns:
        StreamContinuationService 实例
    """
    global _stream_continuation_service
    if _stream_continuation_service is None:
        _stream_continuation_service = StreamContinuationService()
    return _stream_continuation_service


__all__ = [
    "StreamEvent",
    "StreamMetadata",
    "StreamContinuationService",
    "get_stream_continuation_service",
]
