"""流继续管理器

提供活跃流式会话注册表和客户端重连支持。
对齐 WeKnora 的 sessions/continue-stream/{id} 实现。

核心功能：
- ActiveStreamRegistry: 跟踪活跃流
- 支持客户端断线重连
- 流事件缓冲和回放
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.observability.logging import get_logger
from app.agent.streaming.service import (
    StreamEvent as StreamContinuationEvent,
)
from app.agent.streaming.service import (
    get_stream_continuation_service,
)

logger = get_logger(__name__)


class ActiveStreamRegistry:
    """活跃流注册表

    跟踪当前正在执行的流式会话，支持：
    - 注册新的流式会话
    - 检查会话是否活跃
    - 获取流事件
    - 清理完成的会话

    Examples:
        ```python
        registry = ActiveStreamRegistry()

        # 注册流
        await registry.register("session-123", user_id=1)

        # 检查是否活跃
        is_active = await registry.is_active("session-123")

        # 获取事件
        async for event in registry.get_events("session-123"):
            yield event
        ```
    """

    def __init__(self) -> None:
        """初始化注册表"""
        self._service = get_stream_continuation_service()
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def register(
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
        return await self._service.register_stream(session_id, user_id, tenant_id)

    async def is_active(self, session_id: str) -> bool:
        """检查会话是否活跃

        Args:
            session_id: 会话 ID

        Returns:
            True 表示活跃
        """
        return await self._service.is_stream_active(session_id)

    async def get_events(
        self,
        session_id: str,
        since: int = 0,
    ) -> AsyncIterator[StreamContinuationEvent]:
        """获取会话事件

        Args:
            session_id: 会话 ID
            since: 从第几个事件开始

        Yields:
            StreamContinuationEvent 实例
        """
        async for event in self._service.get_events(session_id, since=since):
            yield event

    async def complete(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """标记会话完成

        Args:
            session_id: 会话 ID
            metadata: 元数据

        Returns:
            是否成功
        """
        success = await self._service.complete_stream(session_id, metadata)

        if session_id in self._active_tasks:
            self._active_tasks[session_id].cancel()
            del self._active_tasks[session_id]

        return success

    async def abort(
        self,
        session_id: str,
        reason: str = "Stream aborted",
    ) -> bool:
        """中止会话

        Args:
            session_id: 会话 ID
            reason: 原因

        Returns:
            是否成功
        """
        return await self._service.abort_stream(session_id, reason)

    async def cleanup(self, session_id: str) -> bool:
        """清理会话

        Args:
            session_id: 会话 ID

        Returns:
            是否成功
        """
        if session_id in self._active_tasks:
            self._active_tasks[session_id].cancel()
            del self._active_tasks[session_id]

        return await self._service.cleanup_stream(session_id)


class ContinuableStreamProcessor:
    """可继续的流式处理器

    包装 LangGraph 流式输出，支持客户端重连。

    Examples:
        ```python
        graph = await agent.get_compiled_graph()
        processor = ContinuableStreamProcessor(graph, "session-123")

        # 开始流式处理
        async for event in processor.stream_events(input_data, config):
            print(event)
        ```

        客户端断线后可以重连：
        ```python
        # 继续接收流
        async for event in processor.continue_stream():
            print(event)
        ```
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        session_id: str,
        user_id: int | None = None,
        tenant_id: int | None = None,
    ) -> None:
        """初始化可继续流处理器

        Args:
            graph: 编译后的 LangGraph
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID
        """
        self._graph = graph
        self._session_id = session_id
        self._user_id = user_id
        self._tenant_id = tenant_id
        self._registry = ActiveStreamRegistry()
        self._service = get_stream_continuation_service()

    async def _emit_event(self, event: StreamContinuationEvent) -> None:
        """发送事件到注册表

        Args:
            event: 流事件
        """
        await self._service.add_event(self._session_id, event)

    async def stream_events(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
        version: str = "v1",
    ) -> AsyncIterator[StreamContinuationEvent]:
        """流式输出事件（可继续）

        Args:
            input_data: 输入数据
            config: 运行配置
            version: LangGraph 事件版本

        Yields:
            StreamContinuationEvent 实例
        """
        # 注册流
        await self._registry.register(
            self._session_id,
            self._user_id,
            self._tenant_id,
        )

        logger.info(
            "continuable_stream_start",
            session_id=self._session_id,
        )

        try:
            async for event in self._graph.astream_events(
                input_data,
                config,
                version=version,
            ):
                # 解析 LangGraph 事件
                event_type = event.get("event", "")
                data = event.get("data", {})
                metadata = event.get("metadata", {})

                # Token 事件
                if "on_chat_model_stream" in event_type:
                    token = data.get("chunk", "")
                    if hasattr(token, "content"):
                        content = token.content
                        if isinstance(content, str):
                            stream_event = StreamContinuationEvent(
                                event_type="token",
                                content=content,
                                metadata={"langgraph_event": event_type},
                            )
                            await self._emit_event(stream_event)
                            yield stream_event
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str):
                                    stream_event = StreamContinuationEvent(
                                        event_type="token",
                                        content=item,
                                        metadata={"langgraph_event": event_type},
                                    )
                                    await self._emit_event(stream_event)
                                    yield stream_event

                # 工具调用开始
                elif "on_tool_start" in event_type:
                    tool_name = metadata.get("langgraph_tool_name", "unknown")
                    stream_event = StreamContinuationEvent(
                        event_type="tool_start",
                        content=f"[调用工具: {tool_name}]",
                        metadata={
                            "tool_name": tool_name,
                            "input": data.get("input", {}),
                        },
                    )
                    await self._emit_event(stream_event)
                    yield stream_event

                # 工具调用结束
                elif "on_tool_end" in event_type:
                    tool_name = metadata.get("langgraph_tool_name", "unknown")
                    stream_event = StreamContinuationEvent(
                        event_type="tool_end",
                        content=f"[工具完成: {tool_name}]",
                        metadata={
                            "tool_name": tool_name,
                            "output": data.get("output", {}),
                        },
                    )
                    await self._emit_event(stream_event)
                    yield stream_event

                # 错误事件
                elif "error" in event_type.lower():
                    error_msg = str(data.get("error", "Unknown error"))
                    stream_event = StreamContinuationEvent(
                        event_type="error",
                        content=f"[错误: {error_msg}]",
                        metadata={"error": error_msg},
                    )
                    await self._emit_event(stream_event)
                    yield stream_event

            # 发送完成事件
            done_event = StreamContinuationEvent(
                event_type="done",
                content="",
                metadata={"completed": True},
            )
            await self._emit_event(done_event)
            yield done_event

            # 标记流完成
            await self._registry.complete(self._session_id)

            logger.info(
                "continuable_stream_complete",
                session_id=self._session_id,
            )

        except Exception as e:
            logger.exception(
                "continuable_stream_failed",
                session_id=self._session_id,
                error=str(e),
            )

            # 发送错误事件
            error_event = StreamContinuationEvent(
                event_type="error",
                content=str(e),
                metadata={"error": str(e), "exception": True},
            )
            await self._emit_event(error_event)
            yield error_event

            # 标记流完成（带错误）
            await self._registry.complete(
                self._session_id,
                metadata={"error": str(e)},
            )

    async def continue_stream(
        self,
        since: int = 0,
    ) -> AsyncIterator[StreamContinuationEvent]:
        """继续接收流（用于客户端重连）

        Args:
            since: 从第几个事件开始

        Yields:
            StreamContinuationEvent 实例
        """
        logger.info(
            "continue_stream",
            session_id=self._session_id,
            since=since,
        )

        # 检查流是否活跃
        is_active = await self._registry.is_active(self._session_id)

        if not is_active:
            # 流已完成，返回缓存的事件
            metadata = await self._service.get_metadata(self._session_id)
            if metadata and not metadata.is_active:
                async for event in self._service.get_events(self._session_id, since=since):
                    yield event
                return

        # 流仍然活跃，实时获取事件
        # 首先发送缓存的事件
        async for event in self._service.get_events(self._session_id, since=since):
            yield event

        # 等待新事件（轮询模式）
        last_count = 0
        while await self._registry.is_active(self._session_id):
            events = await self._service.get_latest_events(self._session_id, count=100)
            if len(events) > last_count:
                for event in events[last_count:]:
                    yield event
                last_count = len(events)

            # 检查是否有完成事件
            if events and events[-1].event_type == "done":
                break

            await asyncio.sleep(0.1)

        logger.info(
            "continue_stream_complete",
            session_id=self._session_id,
        )

    async def stream_tokens(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[StreamContinuationEvent]:
        """流式输出 token（可继续）

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            StreamContinuationEvent 实例
        """
        # 注册流
        await self._registry.register(
            self._session_id,
            self._user_id,
            self._tenant_id,
        )

        logger.info(
            "continuable_tokens_start",
            session_id=self._session_id,
        )

        try:
            async for chunk, metadata in self._graph.astream(
                input_data,
                config,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        stream_event = StreamContinuationEvent(
                            event_type="token",
                            content=chunk.content,
                            metadata={
                                "langgraph_node": metadata.get("langgraph_node"),
                                "run_id": metadata.get("run_id"),
                            },
                        )
                        await self._emit_event(stream_event)
                        yield stream_event

            # 发送完成事件
            done_event = StreamContinuationEvent(
                event_type="done",
                content="",
                metadata={"completed": True},
            )
            await self._emit_event(done_event)
            yield done_event

            # 标记流完成
            await self._registry.complete(self._session_id)

            logger.info(
                "continuable_tokens_complete",
                session_id=self._session_id,
            )

        except Exception as e:
            logger.exception(
                "continuable_tokens_failed",
                session_id=self._session_id,
                error=str(e),
            )

            # 发送错误事件
            error_event = StreamContinuationEvent(
                event_type="error",
                content=str(e),
                metadata={"error": str(e), "exception": True},
            )
            await self._emit_event(error_event)
            yield error_event

            # 标记流完成（带错误）
            await self._registry.complete(
                self._session_id,
                metadata={"error": str(e)},
            )


# 全局实例
_active_stream_registry: ActiveStreamRegistry | None = None


def get_active_stream_registry() -> ActiveStreamRegistry:
    """获取活跃流注册表实例（单例）

    Returns:
        ActiveStreamRegistry 实例
    """
    global _active_stream_registry
    if _active_stream_registry is None:
        _active_stream_registry = ActiveStreamRegistry()
    return _active_stream_registry


__all__ = [
    "ActiveStreamRegistry",
    "ContinuableStreamProcessor",
    "get_active_stream_registry",
]
