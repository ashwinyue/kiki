"""运行时配置

参考 DeerFlow 的 Configuration 设计，支持运行时配置驱动。

使用示例:
```python
from app.core.configuration import Configuration

# 从环境变量创建
config = Configuration.from_env()

# 从 RunnableConfig 创建（用于 LangGraph）
config = Configuration.from_runnable_config(runnable_config)

# 访问配置
max_iterations = config.max_iterations
interrupt_tools = config.interrupt_before_tools
```
"""

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.config import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass(kw_only=True)
class Configuration:
    """运行时配置

    参考 DeerFlow 的 Configuration dataclass 设计。
    支持从环境变量和 RunnableConfig 创建配置。

    Attributes:
        max_iterations: 最大迭代次数
        max_step_num: 最大步骤数
        interrupt_before_tools: 需要中断执行的工具名称列表
        enable_clarification: 是否启用澄清功能
        locale: 语言环境
        metadata: 扩展元数据
    """

    max_iterations: int = 50
    max_step_num: int = 10
    interrupt_before_tools: list[str] = field(default_factory=list)
    enable_clarification: bool = False
    locale: str = "zh-CN"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """从 RunnableConfig 创建配置

        Args:
            config: LangGraph RunnableConfig

        Returns:
            Configuration 实例

        Examples:
            >>> from langgraph.types import RunnableConfig
            >>> config = RunnableConfig(
            ...     configurable={"max_iterations": 100}
            ... )
            >>> runtime_config = Configuration.from_runnable_config(config)
        """
        if config is None:
            return cls()

        configurable = config.get("configurable", {})

        # 从 configurable 中读取配置
        values = {
            "max_iterations": configurable.get(
                "max_iterations",
                int(os.environ.get("KIKI_AGENT_MAX_ITERATIONS", settings.agent_max_iterations)),
            ),
            "max_step_num": configurable.get(
                "max_step_num",
                int(os.environ.get("KIKI_AGENT_MAX_STEP_NUM", 10)),
            ),
            "interrupt_before_tools": configurable.get(
                "interrupt_before_tools",
                os.environ.get("KIKI_INTERRUPT_BEFORE_TOOLS", "").split(",") if os.environ.get("KIKI_INTERRUPT_BEFORE_TOOLS") else [],
            ),
            "enable_clarification": configurable.get(
                "enable_clarification",
                os.environ.get("KIKI_ENABLE_CLARIFICATION", "false").lower() == "true",
            ),
            "locale": configurable.get(
                "locale",
                os.environ.get("KIKI_LOCALE", "zh-CN"),
            ),
            "metadata": configurable.get("metadata", {}),
        }

        return cls(**values)

    @classmethod
    def from_env(cls) -> "Configuration":
        """从环境变量创建配置

        Returns:
            Configuration 实例
        """
        return cls(
            max_iterations=int(os.environ.get("KIKI_AGENT_MAX_ITERATIONS", settings.agent_max_iterations)),
            max_step_num=int(os.environ.get("KIKI_AGENT_MAX_STEP_NUM", 10)),
            interrupt_before_tools=os.environ.get("KIKI_INTERRUPT_BEFORE_TOOLS", "").split(",") if os.environ.get("KIKI_INTERRUPT_BEFORE_TOOLS") else [],
            enable_clarification=os.environ.get("KIKI_ENABLE_CLARIFICATION", "false").lower() == "true",
            locale=os.environ.get("KIKI_LOCALE", "zh-CN"),
        )

    def with_overrides(self, **kwargs) -> "Configuration":
        """返回新的配置实例，覆盖指定字段

        Args:
            **kwargs: 要覆盖的字段

        Returns:
            新的 Configuration 实例
        """
        return Configuration(
            max_iterations=kwargs.get("max_iterations", self.max_iterations),
            max_step_num=kwargs.get("max_step_num", self.max_step_num),
            interrupt_before_tools=kwargs.get("interrupt_before_tools", self.interrupt_before_tools.copy()),
            enable_clarification=kwargs.get("enable_clarification", self.enable_clarification),
            locale=kwargs.get("locale", self.locale),
            metadata={**self.metadata, **kwargs.get("metadata", {})},
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Returns:
            配置字典
        """
        return {
            "max_iterations": self.max_iterations,
            "max_step_num": self.max_step_num,
            "interrupt_before_tools": self.interrupt_before_tools.copy(),
            "enable_clarification": self.enable_clarification,
            "locale": self.locale,
            "metadata": self.metadata.copy(),
        }


@dataclass(kw_only=True)
class AgentRuntimeConfig(Configuration):
    """Agent 运行时配置

    继承自 Configuration，添加 Agent 特定的配置项。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型
        system_prompt: 系统提示词
        temperature: LLM 温度
        max_tokens: 最大 token 数
    """

    agent_id: str | None = None
    agent_type: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "AgentRuntimeConfig":
        """从 RunnableConfig 创建配置

        Args:
            config: LangGraph RunnableConfig

        Returns:
            AgentRuntimeConfig 实例
        """
        base_config = Configuration.from_runnable_config(config)

        if config is None:
            return cls(**base_config.to_dict())

        configurable = config.get("configurable", {})

        return cls(
            **base_config.to_dict(),
            agent_id=configurable.get("agent_id"),
            agent_type=configurable.get("agent_type"),
            system_prompt=configurable.get("system_prompt"),
            temperature=float(configurable.get("temperature", settings.llm_temperature)),
            max_tokens=int(configurable.get("max_tokens")) if configurable.get("max_tokens") else settings.llm_max_tokens,
        )


def get_runtime_config(config: RunnableConfig | None = None) -> Configuration:
    """获取运行时配置的便捷函数

    Args:
        config: LangGraph RunnableConfig

    Returns:
        Configuration 实例
    """
    return Configuration.from_runnable_config(config)


def get_agent_runtime_config(config: RunnableConfig | None = None) -> AgentRuntimeConfig:
    """获取 Agent 运行时配置的便捷函数

    Args:
        config: LangGraph RunnableConfig

    Returns:
        AgentRuntimeConfig 实例
    """
    return AgentRuntimeConfig.from_runnable_config(config)
