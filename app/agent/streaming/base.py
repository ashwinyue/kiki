"""LangGraph 流式处理工具

提供统一的流式输出接口，支持多种流模式。
"""

from collections.abc import AsyncIterator
from typing import Any, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.observability.logging import get_logger

logger = get_logger(__name__)


class MessageTuple(NamedTuple):
    """消息元组（用于 messages-tuple 流模式）

    Attributes:
        content: token 内容
        is_last: 是否是最后一个 token
    """
    content: str
    is_last: bool


class StreamEvent:
    """流事件

    Attributes:
        type: 事件类型 (token, tool_start, tool_end, error, metadata)
        content: 事件内容
        metadata: 额外元数据
    """

    TOKEN = "token"  # noqa: S105
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ERROR = "error"
    METADATA = "metadata"

    def __init__(
        self,
        event_type: str,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.type = event_type
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
        }


class StreamProcessor:
    """LangGraph 流式处理器

    提供统一的流式输出接口，支持：
    - token 级别的 LLM 输出
    - 工具调用事件
    - 错误处理
    - 元数据更新
    """

    def __init__(self, graph: CompiledStateGraph):
        """初始化流处理器

        Args:
            graph: 编译后的 LangGraph
        """
        self._graph = graph

    async def stream_tokens(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[str]:
        """流式输出 token

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            token 文本片段
        """
        logger.info("stream_tokens_start")

        async for chunk in self._graph.astream(
            input_data,
            config,
            stream_mode="messages",
        ):
            # 处理消息片段
            if isinstance(chunk, (AIMessage, HumanMessage, ToolMessage)):
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        # 处理多模态内容
                        for item in chunk.content:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict) and "text" in item:
                                yield item["text"]

        logger.info("stream_tokens_complete")

    async def stream_events(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
        version: str = "v2",
    ) -> AsyncIterator[StreamEvent]:
        """流式输出事件

        提供更详细的流式事件，包括：
        - token 事件
        - 工具调用事件
        - 错误事件
        - 元数据更新

        Args:
            input_data: 输入数据
            config: 运行配置
            version: LangGraph 事件版本 (v1, v2, v2_patches)

        Yields:
            StreamEvent 实例
        """
        logger.info("stream_events_start", version=version)

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
                        yield StreamEvent(
                            StreamEvent.TOKEN,
                            content=content,
                            metadata={"langgraph_event": event_type},
                        )
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                yield StreamEvent(
                                    StreamEvent.TOKEN,
                                    content=item,
                                    metadata={"langgraph_event": event_type},
                                )

            # 工具调用开始
            elif "on_tool_start" in event_type:
                tool_name = metadata.get("langgraph_tool_name", "unknown")
                yield StreamEvent(
                    StreamEvent.TOOL_START,
                    content=f"[调用工具: {tool_name}]",
                    metadata={
                        "tool_name": tool_name,
                        "input": data.get("input", {}),
                    },
                )

            # 工具调用结束
            elif "on_tool_end" in event_type:
                tool_name = metadata.get("langgraph_tool_name", "unknown")
                yield StreamEvent(
                    StreamEvent.TOOL_END,
                    content=f"[工具完成: {tool_name}]",
                    metadata={
                        "tool_name": tool_name,
                        "output": data.get("output", {}),
                    },
                )

            # 错误事件
            elif "error" in event_type.lower():
                error_msg = str(data.get("error", "Unknown error"))
                yield StreamEvent(
                    StreamEvent.ERROR,
                    content=f"[错误: {error_msg}]",
                    metadata={"error": error_msg},
                )

        logger.info("stream_events_complete")

    async def stream_updates(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式输出状态更新

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            状态更新字典
        """
        logger.info("stream_updates_start")

        async for update in self._graph.astream(
            input_data,
            config,
            stream_mode="updates",
        ):
            if isinstance(update, dict):
                yield update

        logger.info("stream_updates_complete")

    async def stream_values(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式输出状态值

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            完整状态值
        """
        logger.info("stream_values_start")

        async for value in self._graph.astream(
            input_data,
            config,
            stream_mode="values",
        ):
            if isinstance(value, dict):
                yield value

        logger.info("stream_values_complete")

    async def stream_messages_tuple(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[MessageTuple]:
        """流式输出消息元组 (token, is_last)

        新版 LangGraph 的 messages-tuple 模式，提供更精细的控制。

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            MessageTuple 元组 (token 内容, 是否最后一个)
        """
        logger.info("stream_messages_tuple_start")

        try:
            async for chunk in self._graph.astream(
                input_data,
                config,
                stream_mode="messages-tuple",
            ):
                # messages-tuple 模式返回 (BaseMessage, bool) 元组
                if isinstance(chunk, tuple):
                    message, is_last = chunk
                    if hasattr(message, "content") and message.content:
                        content = message.content
                        if isinstance(content, str):
                            yield MessageTuple(content=content, is_last=bool(is_last))
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str):
                                    yield MessageTuple(content=item, is_last=bool(is_last))
        except Exception as e:
            logger.error("stream_messages_tuple_error", error=str(e))
            raise

        logger.info("stream_messages_tuple_complete")

    async def stream_debug(
        self,
        input_data: dict[str, Any],
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式输出调试信息

        使用 LangGraph 的 debug 流模式，返回完整的执行轨迹，
        包括节点执行、状态更新、工具调用等详细信息。

        Args:
            input_data: 输入数据
            config: 运行配置

        Yields:
            调试事件字典，包含：
            - event: 事件类型
            - node: 节点名称
            - state: 状态快照
            - metadata: 额外信息
        """
        logger.info("stream_debug_start")

        try:
            async for debug_event in self._graph.astream(
                input_data,
                config,
                stream_mode="debug",
            ):
                if isinstance(debug_event, dict):
                    yield debug_event

        except Exception as e:
            logger.error("stream_debug_error", error=str(e))
            raise

        logger.info("stream_debug_complete")


async def stream_tokens_from_graph(
    graph: CompiledStateGraph,
    input_data: dict[str, Any],
    config: RunnableConfig,
) -> AsyncIterator[str]:
    """便捷函数：从图中流式获取 token

    Args:
        graph: 编译后的 LangGraph
        input_data: 输入数据
        config: 运行配置

    Yields:
        token 文本片段

    Examples:
        ```python
        from app.agent.streaming import stream_tokens_from_graph

        async for token in stream_tokens_from_graph(graph, input_data, config):
            print(token, end="")
        ```
    """
    processor = StreamProcessor(graph)
    async for token in processor.stream_tokens(input_data, config):
        yield token


async def stream_events_from_graph(
    graph: CompiledStateGraph,
    input_data: dict[str, Any],
    config: RunnableConfig,
    version: str = "v2",
) -> AsyncIterator[StreamEvent]:
    """便捷函数：从图中流式获取事件

    Args:
        graph: 编译后的 LangGraph
        input_data: 输入数据
        config: 运行配置
        version: LangGraph 事件版本 (v1, v2, v2_patches)

    Yields:
        StreamEvent 实例

    Examples:
        ```python
        from app.agent.streaming import stream_events_from_graph

        async for event in stream_events_from_graph(graph, input_data, config):
            if event.type == StreamEvent.TOKEN:
                print(event.content, end="")
            elif event.type == StreamEvent.TOOL_START:
                print(f"\\n[调用工具: {event.metadata['tool_name']}]")
        ```
    """
    processor = StreamProcessor(graph)
    async for event in processor.stream_events(input_data, config, version):
        yield event


async def stream_messages_tuple_from_graph(
    graph: CompiledStateGraph,
    input_data: dict[str, Any],
    config: RunnableConfig,
) -> AsyncIterator[MessageTuple]:
    """便捷函数：从图中流式获取消息元组 (token, is_last)

    Args:
        graph: 编译后的 LangGraph
        input_data: 输入数据
        config: 运行配置

    Yields:
        MessageTuple 元组

    Examples:
        ```python
        from app.agent.streaming import stream_messages_tuple_from_graph

        async for msg in stream_messages_tuple_from_graph(graph, input_data, config):
            print(msg.content, end="", flush=True)
            if msg.is_last:
                print("\\n--- Message Complete ---")
        ```
    """
    processor = StreamProcessor(graph)
    async for msg in processor.stream_messages_tuple(input_data, config):
        yield msg


async def stream_debug_from_graph(
    graph: CompiledStateGraph,
    input_data: dict[str, Any],
    config: RunnableConfig,
) -> AsyncIterator[dict[str, Any]]:
    """便捷函数：从图中流式获取调试信息

    Args:
        graph: 编译后的 LangGraph
        input_data: 输入数据
        config: 运行配置

    Yields:
        调试事件字典

    Examples:
        ```python
        from app.agent.streaming import stream_debug_from_graph

        async for debug_event in stream_debug_from_graph(graph, input_data, config):
            print(f"Node: {debug_event.get('node')}")
            print(f"Event: {debug_event.get('event')}")
        ```
    """
    processor = StreamProcessor(graph)
    async for debug_event in processor.stream_debug(input_data, config):
        yield debug_event


__all__ = [
    "StreamEvent",
    "StreamProcessor",
    "MessageTuple",
    "stream_tokens_from_graph",
    "stream_events_from_graph",
    "stream_messages_tuple_from_graph",
    "stream_debug_from_graph",
]
