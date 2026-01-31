"""流式输出模块

提供 SSE (Server-Sent Events) 流式响应支持，提升用户体验。
支持 Agent 流式输出和工具调用进度显示。

使用示例:
```python
from app.agent.streaming import (
    StreamingAgent,
    stream_agent_response,
    TokenStream,
)

# 流式执行 Agent
async for chunk in stream_agent_response(agent, input_data):
    print(chunk, end="")

# 使用 TokenStream
stream = TokenStream()
await stream.add("Hello")
await stream.add("World")
await stream.close()
```
"""

import asyncio
import json
import logging
from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Callable,
)

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langgraph.types import RunnableConfig

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 事件类型 ==============

class StreamEvent:
    """流式事件基类"""

    event_type: str
    content: str | dict | None

    def __init__(self, event_type: str, content: str | dict | None = None):
        self.event_type = event_type
        self.content = content

    def to_sse(self) -> str:
        """转换为 SSE 格式

        Returns:
            SSE 格式字符串
        """
        data = self.content if isinstance(self.content, dict) else {"content": self.content}
        return f"event: {self.event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def __str__(self) -> str:
        return self.to_sse()


class TokenEvent(StreamEvent):
    """Token 事件"""

    def __init__(self, token: str, is_first: bool = False, is_last: bool = False):
        super().__init__("token", {"token": token, "is_first": is_first, "is_last": is_last})
        self.token = token
        self.is_first = is_first
        self.is_last = is_last


class ToolCallEvent(StreamEvent):
    """工具调用事件"""

    def __init__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        status: str = "start",  # start, progress, complete, error
    ):
        super().__init__(
            "tool_call",
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": status,
            },
        )
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.status = status


class ErrorEvent(StreamEvent):
    """错误事件"""

    def __init__(self, error: str, error_type: str = "runtime"):
        super().__init__("error", {"error": error, "error_type": error_type})
        self.error = error
        self.error_type = error_type


class StatusEvent(StreamEvent):
    """状态事件"""

    def __init__(self, status: str, message: str | None = None):
        super().__init__("status", {"status": status, "message": message})
        self.status = status
        self.message = message


class DoneEvent(StreamEvent):
    """完成事件"""

    def __init__(self, final_output: str | None = None):
        super().__init__("done", {"final_output": final_output})
        self.final_output = final_output


# ============== Token Stream ==============

class TokenStream:
    """Token 流

    管理 Token 流式输出，支持缓冲和分批发送。
    """

    def __init__(
        self,
        buffer_size: int = 10,
        flush_interval: float = 0.1,
    ):
        """初始化 Token 流

        Args:
            buffer_size: 缓冲区大小
            flush_interval: 刷新间隔（秒）
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer: deque[str] = deque(maxlen=buffer_size)
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._is_closed = False
        self._token_count = 0
        self._lock = asyncio.Lock()

    async def add(self, token: str) -> None:
        """添加 Token

        Args:
            token: Token 字符串
        """
        if self._is_closed:
            logger.warning("token_stream_closed")
            return

        async with self._lock:
            self._token_count += 1
            is_first = (self._token_count == 1)

            # 创建 Token 事件
            event = TokenEvent(token, is_first=is_first)

            # 如果缓冲区满了，刷新
            if len(self._buffer) >= self.buffer_size:
                await self._flush()

            self._buffer.append(token)

    async def add_event(self, event: StreamEvent) -> None:
        """添加事件

        Args:
            event: 流事件
        """
        if self._is_closed:
            logger.warning("token_stream_closed")
            return

        await self._queue.put(event)

    async def _flush(self) -> None:
        """刷新缓冲区"""
        if not self._buffer:
            return

        content = "".join(self._buffer)
        self._buffer.clear()

        event = TokenEvent(content, is_last=False)
        await self._queue.put(event)

    async def close(self, final_output: str | None = None) -> None:
        """关闭流

        Args:
            final_output: 最终输出
        """
        async with self._lock:
            if self._buffer:
                await self._flush()

            # 发送最后一个 Token 事件
            await self._queue.put(TokenEvent("", is_last=True))

            # 发送完成事件
            await self._queue.put(DoneEvent(final_output))

            self._is_closed = True

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """异步迭代器

        Yields:
            StreamEvent 实例
        """
        while True:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self.flush_interval,
                )
                yield event

                if isinstance(event, DoneEvent):
                    break

            except asyncio.TimeoutError:
                # 超时则刷新缓冲区
                await self._flush()

    async def iter_sse(self) -> AsyncIterator[str]:
        """迭代 SSE 格式输出

        Yields:
            SSE 格式字符串
        """
        async for event in self:
            yield event.to_sse()


# ============== 流式 Agent ==============

class StreamingAgent:
    """流式 Agent

    包装 LangGraph Agent，提供流式输出能力。
    """

    def __init__(
        self,
        graph,
        config: RunnableConfig | None = None,
    ):
        """初始化流式 Agent

        Args:
            graph: LangGraph 编译后的图
            config: 运行配置
        """
        self.graph = graph
        self.config = config or {}
        self._stream = TokenStream()

    async def astream(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """异步流式执行

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            StreamEvent 实例
        """
        merged_config = {**self.config, **(config or {})}

        # 发送开始事件
        yield StatusEvent("start", "开始处理")

        try:
            # 使用 LangGraph 的 astream
            async for chunk in self.graph.astream(input_data, merged_config):
                # 处理不同类型的 chunk
                if isinstance(chunk, dict):
                    yield await self._process_chunk(chunk)

                elif isinstance(chunk, BaseMessage):
                    yield await self._process_message(chunk)

        except Exception as e:
            logger.error("streaming_agent_error", error=str(e))
            yield ErrorEvent(str(e))

        finally:
            # 发送完成事件
            yield StatusEvent("complete", "处理完成")

    async def _process_chunk(self, chunk: dict[str, Any]) -> StreamEvent:
        """处理数据块

        Args:
            chunk: 数据块

        Returns:
            StreamEvent 实例
        """
        # 检查是否有消息
        if "messages" in chunk:
            messages = chunk["messages"]
            if messages:
                last_message = messages[-1]

                if isinstance(last_message, AIMessage):
                    # 检查是否有工具调用
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            yield ToolCallEvent(
                                tool_call.get("name", "unknown"),
                                tool_call.get("args", {}),
                                status="start",
                            )

                elif isinstance(last_message, ToolMessage):
                    # 工具执行结果
                    yield ToolCallEvent(
                        last_message.name,
                        {},
                        status="complete",
                    )

    async def _process_message(self, message: BaseMessage) -> StreamEvent:
        """处理消息

        Args:
            message: 消息

        Returns:
            StreamEvent 实例
        """
        if isinstance(message, AIMessageChunk):
            # 获取内容
            content = message.content

            if isinstance(content, str):
                return TokenEvent(content)

            elif isinstance(content, list):
                # 处理多模态内容
                text_parts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and "text" in c
                ]
                return TokenEvent("".join(text_parts))

        return StatusEvent("message_processed")

    async def astream_sse(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[str]:
        """异步流式执行（SSE 格式）

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            SSE 格式字符串
        """
        async for event in self.astream(input_data, config):
            yield event.to_sse()


# ============== 便捷函数 ==============

async def stream_agent_response(
    graph,
    input_data: dict[str, Any],
    config: RunnableConfig | None = None,
) -> AsyncIterator[str]:
    """流式执行 Agent 并返回 Token

    Args:
        graph: LangGraph 编译后的图
        input_data: 输入数据
        config: 运行配置

    Yields:
        Token 字符串

    Examples:
        ```python
        async for token in stream_agent_response(agent, {"messages": [user_message]}):
            print(token, end="")
        ```
    """
    agent = StreamingAgent(graph, config)

    async for event in agent.astream(input_data):
        if isinstance(event, TokenEvent) and event.token:
            yield event.token


async def stream_agent_sse(
    graph,
    input_data: dict[str, Any],
    config: RunnableConfig | None = None,
) -> AsyncIterator[str]:
    """流式执行 Agent 并返回 SSE 格式

    Args:
        graph: LangGraph 编译后的图
        input_data: 输入数据
        config: 运行配置

    Yields:
        SSE 格式字符串

    Examples:
        ```python
        from fastapi.responses import StreamingResponse

        async def generate():
            async for sse in stream_agent_sse(agent, input_data):
                yield sse

        return StreamingResponse(generate(), media_type="text/event-stream")
        ```
    """
    agent = StreamingAgent(graph, config)

    async for event in agent.astream(input_data):
        yield event.to_sse()


async def collect_stream(
    async_iterator: AsyncIterator[str],
    on_token: Callable[[str], None] | None = None,
) -> str:
    """收集流式输出为完整字符串

    Args:
        async_iterator: 异步迭代器
        on_token: Token 回调函数

    Returns:
        完整字符串

    Examples:
        ```python
        result = await collect_stream(
            stream_agent_response(agent, input_data),
            on_token=lambda t: print(t, end="")
        )
        print(f"\\n完整结果: {result}")
        ```
    """
    result_parts = []

    async for chunk in async_iterator:
        result_parts.append(chunk)
        if on_token:
            on_token(chunk)

    return "".join(result_parts)


# ============== FastAPI 集成 ==============

async def create_fastapi_streaming_response(
    graph,
    input_data: dict[str, Any],
    config: RunnableConfig | None = None,
) -> "StreamingResponse":
    """创建 FastAPI 流式响应

    Args:
        graph: LangGraph 编译后的图
        input_data: 输入数据
        config: 运行配置

    Returns:
        FastAPI StreamingResponse

    Examples:
        ```python
        from fastapi import FastAPI
        from app.agent.streaming import create_fastapi_streaming_response

        app = FastAPI()

        @app.post("/chat")
        async def chat(request: ChatRequest):
            return await create_fastapi_streaming_response(
                agent,
                {"messages": [request.message]},
            )
        ```
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        async for sse in stream_agent_sse(graph, input_data, config):
            yield sse

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


__all__ = [
    # 事件类型
    "StreamEvent",
    "TokenEvent",
    "ToolCallEvent",
    "ErrorEvent",
    "StatusEvent",
    "DoneEvent",
    # Token 流
    "TokenStream",
    # 流式 Agent
    "StreamingAgent",
    # 便捷函数
    "stream_agent_response",
    "stream_agent_sse",
    "collect_stream",
    "create_fastapi_streaming_response",
]
