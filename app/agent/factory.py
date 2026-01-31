"""Agent 工厂模式

提供统一的 Agent 创建接口，支持所有 Agent 类型。
参考 DeerFlow 的 create_agent 设计，增强企业级功能。

使用示例:
```python
from app.agent.factory import AgentFactory, AgentType, AgentConfig

# 创建 Chat Agent
chat_agent = AgentFactory.create_agent(
    agent_type=AgentType.CHAT,
    config=AgentConfig(
        system_prompt="你是一个有用的助手",
        llm_type="claude",  # 使用 Claude 模型
    ),
)

# 创建 ReAct Agent（带工具拦截器）
react_agent = AgentFactory.create_agent(
    agent_type=AgentType.REACT,
    tools=[search_tool, delete_tool],
    config=AgentConfig(
        interrupt_before_tools=["delete"],  # 删除工具需要审批
        llm_type="deepseek",  # 使用 DeepSeek 降低成本
    ),
)
```
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.graphs import BaseGraph, ChatGraph
from app.agent.graphs.react import ReactAgent, create_react_agent
from app.agent.multi_agent import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_multi_agent_system,
)
from app.agent.tools.interceptor import wrap_tools_with_interceptor
from app.config.settings import get_settings
from app.llm import LLMService, get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# Agent 类型定义
AgentType = Literal[
    "chat",  # 基础对话 Agent
    "react",  # ReAct 模式 Agent
    "router",  # 路由 Agent
    "supervisor",  # 监督 Agent
    "handoff",  # 切换 Agent
]

# LLM 类型别名
LLMType = Literal[
    "default",  # 使用配置的默认 LLM
    "claude",  # Claude（高质量）
    "deepseek",  # DeepSeek（低成本）
    "ollama",  # 本地 Ollama
]

# Agent 类型到 LLM 类型的映射（参考 DeerFlow 的 AGENT_LLM_MAP）
# 用于根据 Agent 角色自动选择合适的 LLM
AGENT_LLM_MAP: dict[str, LLMType] = {
    "chat": "default",  # 对话使用默认模型
    "react": "default",  # ReAct 使用默认模型
    "router": "default",  # 路由使用默认模型
    "supervisor": "claude",  # 监督者使用 Claude（需要高质量决策）
    "handoff": "default",  # 切换使用默认模型
}


@dataclass
class AgentConfig:
    """Agent 配置

    参考 DeerFlow 的 Configuration 设计，支持运行时配置。

    Attributes:
        system_prompt: 系统提示词
        llm_type: LLM 类型（default/claude/deepseek/ollama）
        locale: 语言环境
        max_iterations: 最大迭代次数
        interrupt_before_tools: 需要中断执行的工具名称列表
        enable_clarification: 是否启用澄清功能
        metadata: 扩展元数据
    """

    system_prompt: str | None = None
    llm_type: LLMType = "default"
    locale: str = "zh-CN"
    max_iterations: int = 50
    interrupt_before_tools: list[str] = field(default_factory=list)
    enable_clarification: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_llm_type(self, llm_type: LLMType) -> "AgentConfig":
        """返回新的配置实例，修改 LLM 类型

        Args:
            llm_type: 新的 LLM 类型

        Returns:
            新的配置实例
        """
        return AgentConfig(
            system_prompt=self.system_prompt,
            llm_type=llm_type,
            locale=self.locale,
            max_iterations=self.max_iterations,
            interrupt_before_tools=self.interrupt_before_tools.copy(),
            enable_clarification=self.enable_clarification,
            metadata=self.metadata.copy(),
        )


class AgentFactoryError(Exception):
    """Agent 工厂错误"""


class AgentFactory:
    """Agent 工厂类

    提供统一的 Agent 创建接口，支持所有 Agent 类型的创建。
    使用工厂模式简化 Agent 的创建和配置。
    参考 DeerFlow 的 create_agent 设计。
    """

    # 默认配置
    _default_llm_service: LLMService | None = None
    _default_checkpointer: BaseCheckpointSaver | None = None

    @classmethod
    def set_default_llm_service(cls, llm_service: LLMService) -> None:
        """设置默认的 LLM 服务

        Args:
            llm_service: LLM 服务实例
        """
        cls._default_llm_service = llm_service
        logger.info("default_llm_service_set")

    @classmethod
    def set_default_checkpointer(cls, checkpointer: BaseCheckpointSaver) -> None:
        """设置默认的检查点保存器

        Args:
            checkpointer: 检查点保存器
        """
        cls._default_checkpointer = checkpointer
        logger.info("default_checkpointer_set")

    @classmethod
    def create_agent(
        cls,
        agent_type: AgentType,
        llm_service: LLMService | None = None,
        system_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        config: AgentConfig | None = None,
        **kwargs,
    ) -> BaseGraph | Any:
        """创建 Agent 实例

        Args:
            agent_type: Agent 类型
            llm_service: LLM 服务实例（默认使用全局实例）
            system_prompt: 系统提示词（优先级低于 config.system_prompt）
            checkpointer: 检查点保存器
            config: Agent 配置
            **kwargs: Agent 特定参数

        Returns:
            Agent 实例

        Raises:
            AgentFactoryError: 创建失败时

        Agent 类型特定参数:
            - chat: 无额外参数
            - react: tools (list) - 工具列表
            - router: agents (dict) - 子 Agent 字典
            - supervisor: workers (dict) - Worker Agent 字典
            - handoff: name (str), tools (list), handoff_targets (list)
        """
        # 合并配置
        if config is None:
            config = AgentConfig()
        if system_prompt:
            config = AgentConfig(
                system_prompt=system_prompt,
                llm_type=config.llm_type,
                locale=config.locale,
                max_iterations=config.max_iterations,
                interrupt_before_tools=config.interrupt_before_tools.copy(),
                enable_clarification=config.enable_clarification,
                metadata=config.metadata.copy(),
            )

        # 使用默认 LLM 服务
        if llm_service is None:
            llm_service = cls._default_llm_service or get_llm_service()

        # 使用默认检查点
        if checkpointer is None:
            checkpointer = cls._default_checkpointer

        # 根据 Agent 类型选择 LLM（参考 DeerFlow）
        mapped_llm_type = AGENT_LLM_MAP.get(agent_type, "default")
        if config.llm_type == "default" and mapped_llm_type != "default":
            # 使用 Agent 类型映射的 LLM
            llm_service = cls._get_llm_service_for_type(mapped_llm_type, llm_service)
        elif config.llm_type != "default":
            # 使用配置指定的 LLM
            llm_service = cls._get_llm_service_for_type(config.llm_type, llm_service)

        logger.info(
            "creating_agent",
            agent_type=agent_type,
            llm_type=config.llm_type,
            has_checkpointer=checkpointer is not None,
            interrupt_tools=len(config.interrupt_before_tools),
        )

        try:
            if agent_type == "chat":
                return cls._create_chat_agent(
                    llm_service,
                    config,
                    checkpointer,
                    **kwargs,
                )
            elif agent_type == "react":
                return cls._create_react_agent(
                    llm_service,
                    config,
                    checkpointer,
                    **kwargs,
                )
            elif agent_type == "router":
                return cls._create_router_agent(
                    llm_service,
                    config,
                    **kwargs,
                )
            elif agent_type == "supervisor":
                return cls._create_supervisor_agent(
                    llm_service,
                    config,
                    **kwargs,
                )
            elif agent_type == "handoff":
                return cls._create_handoff_agent(
                    llm_service,
                    config,
                    **kwargs,
                )
            else:
                raise AgentFactoryError(f"未知的 Agent 类型: {agent_type}")

        except Exception as e:
            logger.error(
                "agent_creation_failed",
                agent_type=agent_type,
                error=str(e),
            )
            raise AgentFactoryError(f"创建 {agent_type} Agent 失败: {e}") from e

    @classmethod
    def _get_llm_service_for_type(
        cls, llm_type: LLMType, default_service: LLMService
    ) -> LLMService:
        """根据 LLM 类型获取 LLM 服务

        Args:
            llm_type: LLM 类型
            default_service: 默认 LLM 服务

        Returns:
            LLM 服务实例
        """
        if llm_type == "default":
            return default_service

        # TODO: 实现不同 LLM 服务的创建逻辑
        # 目前返回默认服务
        return default_service

    @classmethod
    def _create_chat_agent(
        cls,
        llm_service: LLMService,
        config: AgentConfig,
        checkpointer: BaseCheckpointSaver | None = None,
        **kwargs,
    ) -> ChatGraph:
        """创建基础对话 Agent

        Args:
            llm_service: LLM 服务
            config: Agent 配置
            checkpointer: 检查点保存器

        Returns:
            ChatGraph 实例
        """
        graph = ChatGraph(
            llm_service=llm_service,
            system_prompt=config.system_prompt,
        )
        graph.compile(checkpointer=checkpointer)
        return graph

    @classmethod
    def _create_react_agent(
        cls,
        llm_service: LLMService,
        config: AgentConfig,
        checkpointer: BaseCheckpointSaver | None = None,
        **kwargs,
    ) -> ReactAgent:
        """创建 ReAct 模式 Agent

        Args:
            llm_service: LLM 服务
            config: Agent 配置
            checkpointer: 检查点保存器
            **kwargs: tools (list) - 工具列表

        Returns:
            ReactAgent 实例
        """
        tools = kwargs.get("tools", [])

        # 包装工具拦截器（参考 DeerFlow）
        if config.interrupt_before_tools:
            logger.info(
                "wrapping_tools_with_interceptor",
                count=len(config.interrupt_before_tools),
            )
            tools = wrap_tools_with_interceptor(tools, config.interrupt_before_tools)

        return create_react_agent(
            llm_service=llm_service,
            tools=tools,
            system_prompt=config.system_prompt,
            checkpointer=checkpointer,
        )

    @classmethod
    def _create_router_agent(
        cls,
        llm_service: LLMService,
        config: AgentConfig,
        **kwargs,
    ) -> RouterAgent:
        """创建路由 Agent

        Args:
            llm_service: LLM 服务
            config: Agent 配置
            **kwargs: agents (dict) - 子 Agent 字典

        Returns:
            RouterAgent 实例
        """
        agents = kwargs.get("agents", {})
        if not agents:
            raise AgentFactoryError("Router Agent 需要 agents 参数")

        return RouterAgent(
            llm_service=llm_service,
            agents=agents,
            router_prompt=config.system_prompt,
        )

    @classmethod
    def _create_supervisor_agent(
        cls,
        llm_service: LLMService,
        config: AgentConfig,
        **kwargs,
    ) -> SupervisorAgent:
        """创建监督 Agent

        Args:
            llm_service: LLM 服务
            config: Agent 配置
            **kwargs: workers (dict) - Worker Agent 字典

        Returns:
            SupervisorAgent 实例
        """
        workers = kwargs.get("workers", {})
        if not workers:
            raise AgentFactoryError("Supervisor Agent 需要 workers 参数")

        return SupervisorAgent(
            llm_service=llm_service,
            workers=workers,
            supervisor_prompt=config.system_prompt,
        )

    @classmethod
    def _create_handoff_agent(
        cls,
        llm_service: LLMService,
        config: AgentConfig,
        **kwargs,
    ) -> HandoffAgent:
        """创建可切换 Agent

        Args:
            llm_service: LLM 服务
            config: Agent 配置
            **kwargs:
                - name (str): Agent 名称
                - tools (list): 工具列表
                - handoff_targets (list): 可切换的目标列表

        Returns:
            HandoffAgent 实例
        """
        name = kwargs.get("name")
        if not name:
            raise AgentFactoryError("Handoff Agent 需要 name 参数")

        tools = kwargs.get("tools", [])

        # 包装工具拦截器
        if config.interrupt_before_tools:
            tools = wrap_tools_with_interceptor(tools, config.interrupt_before_tools)

        handoff_targets = kwargs.get("handoff_targets", [])

        return HandoffAgent(
            name=name,
            llm_service=llm_service,
            tools=tools,
            handoff_targets=handoff_targets,
            system_prompt=config.system_prompt,
        )

    @classmethod
    def create_multi_agent_system(
        cls,
        mode: Literal["router", "supervisor", "swarm"],
        llm_service: LLMService | None = None,
        config: AgentConfig | None = None,
        **kwargs,
    ) -> Any:
        """创建多 Agent 系统（便捷方法）

        Args:
            mode: 多 Agent 模式
            llm_service: LLM 服务
            config: Agent 配置
            **kwargs: 模式特定参数

        Returns:
            编译后的 StateGraph
        """
        if llm_service is None:
            llm_service = cls._default_llm_service or get_llm_service()

        if config is None:
            config = AgentConfig()

        return create_multi_agent_system(mode=mode, llm_service=llm_service, **kwargs)


# 便捷函数
def create_agent(
    agent_type: AgentType,
    llm_service: LLMService | None = None,
    system_prompt: str | None = None,
    config: AgentConfig | None = None,
    **kwargs,
) -> BaseGraph | Any:
    """创建 Agent 的便捷函数

    Args:
        agent_type: Agent 类型
        llm_service: LLM 服务
        system_prompt: 系统提示词
        config: Agent 配置
        **kwargs: Agent 特定参数

    Returns:
        Agent 实例

    Examples:
        ```python
        from app.agent.factory import create_agent, AgentType, AgentConfig

        # 创建 Chat Agent
        agent = create_agent(AgentType.CHAT)

        # 创建 ReAct Agent（带配置）
        agent = create_agent(
            AgentType.REACT,
            tools=[search_tool],
            config=AgentConfig(
                llm_type="deepseek",
                interrupt_before_tools=["delete"],
            ),
        )
        ```
    """
    return AgentFactory.create_agent(
        agent_type=agent_type,
        llm_service=llm_service,
        system_prompt=system_prompt,
        config=config,
        **kwargs,
    )
