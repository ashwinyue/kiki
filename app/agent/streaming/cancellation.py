"""流式处理取消支持

提供 CancellationToken 类，支持优雅中断 LangGraph 流式响应。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.infra.redis import get_cache
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Redis 键前缀
_SESSION_STOP_PREFIX = "kiki:session:stop:"


class SessionStoppedError(Exception):
    """会话已停止异常

    当检测到停止信号时抛出此异常。
    """

    def __init__(self, session_id: str, message: str = "Session was stopped by user") -> None:
        """初始化异常

        Args:
            session_id: 会话 ID
            message: 错误消息
        """
        self.session_id = session_id
        super().__init__(message)


class CancellationToken:
    """取消令牌

    用于跟踪和检查取消状态。

    Examples:
        ```python
        token = CancellationToken("session-123")

        # 检查是否已取消
        if await token.is_cancelled():
            raise SessionStoppedError("session-123")

        # 请求取消
        await token.cancel()

        # 重置取消状态
        await token.reset()
        ```
    """

    def __init__(self, session_id: str) -> None:
        """初始化取消令牌

        Args:
            session_id: 会话 ID
        """
        self.session_id = session_id
        self._cache = get_cache(key_prefix=_SESSION_STOP_PREFIX)
        self._check_interval = 0.1  # 检查间隔（秒）

    def _make_key(self) -> str:
        """生成 Redis 键"""
        return f"stopped:{self.session_id}"

    async def is_cancelled(self) -> bool:
        """检查是否已取消

        Returns:
            True 表示已取消
        """
        try:
            return await self._cache.exists(self._make_key())
        except Exception as e:
            logger.error("check_cancel_failed", session_id=self.session_id, error=str(e))
            return False

    async def cancel(self) -> bool:
        """请求取消

        Args:
            设置取消标记，ttl=300 秒

        Returns:
            是否设置成功
        """
        try:
            success = await self._cache.set(self._make_key(), "1", ttl=300)
            if success:
                logger.info("session_cancel_marked", session_id=self.session_id)
            return success
        except Exception as e:
            logger.error("mark_cancel_failed", session_id=self.session_id, error=str(e))
            return False

    async def reset(self) -> bool:
        """重置取消状态

        Returns:
            是否重置成功
        """
        try:
            success = await self._cache.delete(self._make_key())
            if success:
                logger.info("session_cancel_reset", session_id=self.session_id)
            return success
        except Exception as e:
            logger.error("reset_cancel_failed", session_id=self.session_id, error=str(e))
            return False

    async def wait_for_cancel(
        self,
        wait_timeout: float | None = None,
    ) -> bool:
        """等待取消信号

        Args:
            wait_timeout: 超时时间（秒），None 表示无限等待

        Returns:
            True 表示收到取消信号，False 表示超时
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            if await self.is_cancelled():
                return True

            if wait_timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= wait_timeout:
                    return False

            await asyncio.sleep(self._check_interval)


class CancellableStreamProcessor:
    """可取消的流式处理器

    包装 LangGraph 流式输出，支持中途取消。

    Examples:
        ```python
        graph = await agent.get_compiled_graph()
        processor = CancellableStreamProcessor(graph, "session-123")

        async for event in processor.stream_events(input_data, config):
            if event.type == "token":
                print(event.content, end="")
        ```
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        session_id: str,
        check_interval: float = 0.1,
    ) -> None:
        """初始化可取消流处理器

        Args:
            graph: 编译后的 LangGraph
            session_id: 会话 ID
            check_interval: 取消检查间隔（秒）
        """
        self._graph = graph
        self._session_id = session_id
        self._token = CancellationToken(session_id)
        self._check_interval = check_interval

    async def _check_cancelled(self) -> None:
        """检查是否已取消，已取消则抛出异常"""
        if await self._token.is_cancelled():
            logger.info("stream_cancelled", session_id=self._session_id)
            raise SessionStoppedError(self._session_id)

    async def stream_tokens(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[str]:
        """流式输出 token（可取消）

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            token 文本片段

        Raises:
            SessionStoppedError: 会话被停止时
        """
        logger.info("cancellable_stream_start", session_id=self._session_id)

        try:
            async for chunk in self._graph.astream(
                input_data,
                config,
                stream_mode="messages",
            ):
                # 检查取消状态
                await self._check_cancelled()

                # 处理消息片段
                if isinstance(chunk, (AIMessage, HumanMessage, ToolMessage)):
                    if hasattr(chunk, "content") and chunk.content:
                        if isinstance(chunk.content, str):
                            yield chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, str):
                                    yield item
                                elif isinstance(item, dict) and "text" in item:
                                    yield item["text"]

            logger.info("cancellable_stream_complete", session_id=self._session_id)

        except SessionStoppedError:
            # 重新抛出停止异常
            raise
        except Exception as e:
            logger.exception("cancellable_stream_failed", session_id=self._session_id, error=str(e))
            raise

    async def stream_events(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
        version: str = "v2",
    ) -> AsyncIterator[dict[str, Any]]:
        """流式输出 LangGraph 事件（可取消）

        Args:
            input_data: 输入数据
            config: 运行配置
            version: LangGraph 事件版本 (v1, v2, v2_patches)

        Yields:
            LangGraph 事件字典

        Raises:
            SessionStoppedError: 会话被停止时
        """
        logger.info("cancellable_events_start", session_id=self._session_id, version=version)

        try:
            async for event in self._graph.astream_events(
                input_data,
                config,
                version=version,
            ):
                # 检查取消状态
                await self._check_cancelled()

                yield event

            logger.info("cancellable_events_complete", session_id=self._session_id)

        except SessionStoppedError:
            # 重新抛出停止异常
            raise
        except Exception as e:
            logger.exception("cancellable_events_failed", session_id=self._session_id, error=str(e))
            raise

    async def stream_updates(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式输出状态更新（可取消）

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            状态更新字典

        Raises:
            SessionStoppedError: 会话被停止时
        """
        logger.info("cancellable_updates_start", session_id=self._session_id)

        try:
            async for update in self._graph.astream(
                input_data,
                config,
                stream_mode="updates",
            ):
                # 检查取消状态
                await self._check_cancelled()

                if isinstance(update, dict):
                    yield update

            logger.info("cancellable_updates_complete", session_id=self._session_id)

        except SessionStoppedError:
            # 重新抛出停止异常
            raise
        except Exception as e:
            logger.exception("cancellable_updates_failed", session_id=self._session_id, error=str(e))
            raise


async def request_session_stop(session_id: str) -> bool:
    """请求停止会话

    在 Redis 中设置停止标记。

    Args:
        session_id: 会话 ID

    Returns:
        是否成功设置停止标记
    """
    token = CancellationToken(session_id)
    success = await token.cancel()

    if success:
        logger.info("session_stop_requested", session_id=session_id)

    return success


async def is_session_stopped(session_id: str) -> bool:
    """检查会话是否已停止

    Args:
        session_id: 会话 ID

    Returns:
        True 表示会话已停止
    """
    token = CancellationToken(session_id)
    return await token.is_cancelled()


async def reset_session_stop(session_id: str) -> bool:
    """重置会话停止状态

    Args:
        session_id: 会话 ID

    Returns:
        是否成功重置
    """
    token = CancellationToken(session_id)
    return await token.reset()


__all__ = [
    "SessionStoppedError",
    "CancellationToken",
    "CancellableStreamProcessor",
    "request_session_stop",
    "is_session_stopped",
    "reset_session_stop",
]
