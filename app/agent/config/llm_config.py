"""分层 LLM 配置

参考 DeerFlow 的分层 LLM 设计，实现：
- LLMType: LLM 类型定义
- AGENT_LLM_MAP: Agent 与 LLM 类型的映射
- LLM_CONFIG: 分层 LLM 配置
- get_llm_by_type(): 根据 LLM 类型获取模型

核心优势：
1. **成本优化**: 简单任务用便宜模型，复杂任务用强模型
2. **性能提升**: 推理任务用 reasoning 模型，代码任务用 code 模型
3. **灵活配置**: 支持覆盖默认配置

使用示例：
    ```python
    from app.agent.config import get_llm_by_type, AGENT_LLM_MAP

    # 获取 Planner 使用的 LLM（reasoning 类型）
    llm_type = AGENT_LLM_MAP["planner"]
    planner_llm = get_llm_by_type(llm_type)

    # 获取 Coder 使用的 LLM（code 类型）
    coder_llm = get_llm_by_type(AGENT_LLM_MAP["coder"])
    ```
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from app.config.settings import get_settings
from app.llm.registry import LLMRegistry
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ============== LLM 类型定义 ==============

LLMType = Literal[
    "reasoning",  # 推理模型（复杂规划、深度分析）
    "basic",      # 基础模型（通用对话、简单任务）
    "vision",     # 视觉模型（图像理解、多模态）
    "code",       # 代码模型（代码生成、代码分析）
]


# ============== Agent-LLM 映射 ==============

AGENT_LLM_MAP: dict[str, LLMType] = {
    # ========== Supervisor 模式 ==========
    "supervisor": "reasoning",  # Supervisor 需要理解全局，使用推理模型
    "router": "basic",          # Router 只需要简单路由，使用基础模型

    # ========== 专门化角色（DeerFlow 风格）==========
    "coordinator": "basic",     # Coordinator 入口任务，基础模型足够
    "planner": "reasoning",     # Planner 需要深度思考和规划
    "researcher": "basic",      # Researcher 信息检索，基础模型
    "analyst": "basic",         # Analyst 数据分析，基础模型
    "coder": "code",            # Coder 代码生成，使用代码模型
    "reporter": "basic",        # Reporter 聚合输出，基础模型

    # ========== 通用角色 ==========
    "chat": "basic",            # 普通对话，基础模型
    "worker": "basic",          # 通用 Worker，基础模型

    # ========== 工具调用型 ==========
    "tools": "basic",           # 工具调用，基础模型
}


# ============== 分层 LLM 配置 ==============

class LLMTierConfig(BaseModel):
    """LLM 分层配置

    Attributes:
        model: 模型名称
        provider: 提供商（可选，默认使用 settings.llm_provider）
        temperature: 温度参数
        max_tokens: 最大 token 数
        api_key_env: API Key 环境变量名（可选）
        base_url: 自定义 base URL（可选）
    """

    model: str
    provider: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    api_key_env: str | None = None
    base_url: str | None = None


# 默认分层配置
# 可以通过环境变量覆盖：KIKI_LLM__REASONING__MODEL="gpt-4o"
LLM_CONFIG: dict[LLMType, LLMTierConfig] = {
    "reasoning": LLMTierConfig(
        model="deepseek-reasoner",  # DeepSeek Reasoner 是性价比很高的推理模型
        provider="deepseek",
        temperature=0.7,
        max_tokens=100000,  # 推理模型需要更大输出空间
    ),
    "basic": LLMTierConfig(
        model="gpt-4o",  # GPT-4o 平衡性能与成本
        provider="openai",
        temperature=0.7,
    ),
    "vision": LLMTierConfig(
        model="gpt-4o",  # GPT-4o 原生支持视觉
        provider="openai",
        temperature=0.7,
    ),
    "code": LLMTierConfig(
        model="claude-sonnet-4-20250514",  # Claude Sonnet 在代码生成方面表现优异
        provider="anthropic",
        temperature=0.3,  # 代码生成需要更低的温度
    ),
}


# ============== LLM 缓存 ==============

_llm_cache: dict[LLMType, BaseChatModel] = {}


def _load_config_from_settings() -> None:
    """从环境变量加载 LLM 配置

    支持的环境变量格式：
    - KIKI_LLM__REASONING__MODEL="gpt-4o"
    - KIKI_LLM__BASIC__PROVIDER="openai"
    - KIKI_LLM__CODE__TEMPERATURE=0.3

    使用双下划线分隔符支持嵌套配置。
    """
    import os

    for llm_type in LLM_CONFIG:
        # 检查是否有该类型的配置
        prefix = f"kiki_llm__{llm_type}__"
        for key, value in os.environ.items():
            if key.lower().startswith(prefix):
                # 提取配置项
                config_key = key[len(prefix):].lower()
                current_config = LLM_CONFIG[llm_type]

                # 更新配置
                if config_key == "model":
                    LLM_CONFIG[llm_type] = current_config.model_copy(update={"model": value})
                elif config_key == "provider":
                    LLM_CONFIG[llm_type] = current_config.model_copy(update={"provider": value})
                elif config_key == "temperature":
                    try:
                        LLM_CONFIG[llm_type] = current_config.model_copy(
                            update={"temperature": float(value)}
                        )
                    except ValueError:
                        logger.warning("invalid_temperature", llm_type=llm_type, value=value)
                elif config_key == "max_tokens":
                    try:
                        LLM_CONFIG[llm_type] = current_config.model_copy(
                            update={"max_tokens": int(value)}
                        )
                    except ValueError:
                        logger.warning("invalid_max_tokens", llm_type=llm_type, value=value)
                elif config_key == "base_url":
                    LLM_CONFIG[llm_type] = current_config.model_copy(update={"base_url": value})


# 初始化时加载环境变量配置
_load_config_from_settings()


def get_llm_by_type(
    llm_type: LLMType,
    **kwargs: object,
) -> BaseChatModel:
    """根据 LLM 类型获取模型

    这是分层 LLM 配置的核心函数。
    支持缓存、配置合并、自动重试。

    Args:
        llm_type: LLM 类型 (reasoning/basic/vision/code)
        **kwargs: 覆盖默认配置的参数

    Returns:
        配置好的 LLM 实例

    Raises:
        ValueError: 如果模型未注册或创建失败

    Examples:
        ```python
        # 获取推理模型
        reasoning_llm = get_llm_by_type("reasoning")

        # 获取代码模型，并覆盖温度参数
        code_llm = get_llm_by_type("code", temperature=0.1)
        ```
    """
    # 检查缓存
    if llm_type in _llm_cache and not kwargs:
        logger.debug("using_cached_llm", llm_type=llm_type)
        return _llm_cache[llm_type]

    # 获取配置
    if llm_type not in LLM_CONFIG:
        available = ", ".join(LLM_CONFIG.keys())
        raise ValueError(
            f"未知的 LLM 类型: '{llm_type}'。可用类型: {available}"
        )

    config = LLM_CONFIG[llm_type]

    # 合并配置
    merged_kwargs: dict[str, object] = {
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        merged_kwargs["max_tokens"] = config.max_tokens
    if config.api_key_env:
        import os
        merged_kwargs["api_key"] = os.getenv(config.api_key_env)
    if config.base_url:
        merged_kwargs["base_url"] = config.base_url

    # 应用用户覆盖
    merged_kwargs.update(kwargs)

    # 获取模型
    try:
        # 优先检查模型是否已在 LLMRegistry 中注册
        if LLMRegistry.is_registered(config.model):
            llm = LLMRegistry.get(config.model, **merged_kwargs)
        else:
            # 如果未注册，尝试动态创建
            logger.info(
                "creating_llm_dynamically",
                llm_type=llm_type,
                model=config.model,
            )

            # 根据提供商创建 LLM
            provider = config.provider or settings.llm_provider

            if provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.model,
                    api_key=merged_kwargs.get("api_key") or settings.llm_api_key,
                    base_url=merged_kwargs.get("base_url") or settings.llm_base_url,
                    temperature=merged_kwargs["temperature"],
                    max_tokens=merged_kwargs.get("max_tokens"),
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=config.model,
                    api_key=merged_kwargs.get("api_key") or settings.llm_api_key,
                    temperature=merged_kwargs["temperature"],
                    max_tokens=merged_kwargs.get("max_tokens"),
                )
            elif provider == "deepseek":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.model,
                    api_key=merged_kwargs.get("api_key") or settings.deepseek_api_key or settings.llm_api_key,
                    base_url=merged_kwargs.get("base_url") or settings.deepseek_base_url,
                    temperature=merged_kwargs["temperature"],
                    max_tokens=merged_kwargs.get("max_tokens"),
                )
            elif provider == "dashscope":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.model,
                    api_key=merged_kwargs.get("api_key") or settings.dashscope_api_key or settings.llm_api_key,
                    base_url=merged_kwargs.get("base_url") or settings.dashscope_base_url,
                    temperature=merged_kwargs["temperature"],
                    max_tokens=merged_kwargs.get("max_tokens"),
                )
            else:
                raise ValueError(f"不支持的提供商: '{provider}'")

        # 缓存 LLM（仅在无自定义参数时）
        if not kwargs:
            _llm_cache[llm_type] = llm

        logger.info(
            "llm_created",
            llm_type=llm_type,
            model=config.model,
            provider=config.provider or settings.llm_provider,
        )
        return llm

    except Exception as e:
        logger.error(
            "llm_creation_failed",
            llm_type=llm_type,
            model=config.model,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


def get_llm_for_agent(agent_type: str, **kwargs: object) -> BaseChatModel:
    """根据 Agent 类型获取对应的 LLM

    这是 Agent 工厂推荐使用的 LLM 获取方法。

    Args:
        agent_type: Agent 类型（如 planner, coder, researcher）
        **kwargs: 覆盖默认配置的参数

    Returns:
        配置好的 LLM 实例

    Raises:
        ValueError: 如果 Agent 类型未映射到 LLM

    Examples:
        ```python
        # 获取 Planner 的 LLM
        planner_llm = get_llm_for_agent("planner")

        # 获取 Coder 的 LLM，并自定义参数
        coder_llm = get_llm_for_agent("coder", temperature=0.1)
        ```
    """
    if agent_type not in AGENT_LLM_MAP:
        available = ", ".join(AGENT_LLM_MAP.keys())
        raise ValueError(
            f"未映射的 Agent 类型: '{agent_type}'。可用类型: {available}"
        )

    llm_type = AGENT_LLM_MAP[agent_type]
    return get_llm_by_type(llm_type, **kwargs)


def clear_llm_cache() -> None:
    """清除 LLM 缓存

    主要用于：
    1. 测试环境重置
    2. 配置更新后重新加载
    """
    global _llm_cache
    _llm_cache.clear()
    logger.info("llm_cache_cleared")


def list_agent_types() -> list[str]:
    """列出所有已映射的 Agent 类型

    Returns:
        Agent 类型列表
    """
    return list(AGENT_LLM_MAP.keys())


def list_llm_types() -> list[LLMType]:
    """列出所有可用的 LLM 类型

    Returns:
        LLM 类型列表
    """
    return list(LLM_CONFIG.keys())
