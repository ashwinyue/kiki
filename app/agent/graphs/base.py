"""图工作流抽象基类

定义所有图工作流必须实现的接口。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from app.agent.state import AgentState


class BaseGraph(ABC):
    """图工作流抽象基类

    所有图工作流必须继承此类并实现 compile 方法。
    提供统一的调用接口：ainvoke、astream、aget_state。

    Attributes:
        _llm_service: LLM 服务实例
        _system_prompt: 系统提示词
        _graph: 编译后的图实例
        _checkpointer: 检查点保存器
    """

    def __init__(
        self,
        llm_service,
        system_prompt: str | None = None,
    ) -> None:
        """初始化图

        Args:
            llm_service: LLM 服务实例
            system_prompt: 系统提示词
        """
        self._llm_service = llm_service
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._graph: CompiledStateGraph | None = None
        self._checkpointer: BaseCheckpointSaver | None = None

    def _default_system_prompt(self) -> str:
        """默认系统提示词

        子类可以覆盖此方法以提供自定义提示词。
        """
        return """你是一个有用的 AI 助手，可以帮助用户解答问题和完成各种任务。

你可以使用提供的工具来获取信息或执行操作。请始终以友好、专业的方式回应用户。

如果用户的问题超出了你的知识范围或工具能力，请诚实地告知用户。"""

    @abstractmethod
    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        """编译图

        子类必须实现此方法，构建并编译 StateGraph。

        Args:
            checkpointer: 检查点保存器

        Returns:
            编译后的 CompiledStateGraph
        """
        pass

    async def ainvoke(
        self,
        input_data: dict,
        config: RunnableConfig,
    ) -> AgentState:
        """异步调用图

        Args:
            input_data: 输入数据
            config: 运行配置

        Returns:
            最终状态
        """
        if self._graph is None:
            self.compile()

        return await self._graph.ainvoke(input_data, config)

    async def astream(
        self,
        input_data: dict,
        config: RunnableConfig,
        stream_mode: str = "messages",
    ) -> AsyncIterator:
        """异步流式调用图

        Args:
            input_data: 输入数据
            config: 运行配置
            stream_mode: 流模式（messages, updates, values）

        Yields:
            流式数据
        """
        if self._graph is None:
            self.compile()

        async for chunk in self._graph.astream(input_data, config, stream_mode=stream_mode):
            yield chunk

    async def astream_events(
        self,
        input_data: dict,
        config: RunnableConfig,
        version: str = "v1",
    ) -> AsyncIterator:
        """异步流式事件

        Args:
            input_data: 输入数据
            config: 运行配置
            version: 版本

        Yields:
            事件数据
        """
        if self._graph is None:
            self.compile()

        async for event in self._graph.astream_events(input_data, config, version=version):
            yield event

    async def aget_state(self, config: RunnableConfig):
        """异步获取当前状态

        Args:
            config: 运行配置

        Returns:
            状态快照
        """
        if self._graph is None:
            self.compile()

        # get_state 是同步方法，不需要 await
        return self._graph.get_state(config)
