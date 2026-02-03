"""多模型提供商路由系统

支持基于优先级（成本/质量/速度）的自动模型选择。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LLMPriority(str, Enum):
    """LLM 选择优先级

    - COST: 成本优先，选择最便宜的模型
    - QUALITY: 质量优先，选择最强推理能力的模型
    - SPEED: 速度优先，选择响应最快的模型
    - BALANCED: 平衡模式，性能与成本折中
    """

    COST = "cost"
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"


@dataclass(frozen=True)
class ModelConfig:
    """模型配置

    Attributes:
        name: 模型名称
        provider: 提供商 (anthropic/openai/deepseek/ollama)
        priority: 适用优先级
        cost_per_1m_tokens: 每 100 万 tokens 成本（美元）
        avg_latency_ms: 平均延迟（毫秒）
        reasoning_score: 推理能力评分 (1-10)
    """

    name: str
    provider: str
    priority: LLMPriority
    cost_per_1m_tokens: float
    avg_latency_ms: int
    reasoning_score: int


_MODEL_REGISTRY: dict[LLMPriority, list[ModelConfig]] = {
    LLMPriority.COST: [
        ModelConfig(
            name="deepseek-chat",
            provider="deepseek",
            priority=LLMPriority.COST,
            cost_per_1m_tokens=0.14,
            avg_latency_ms=800,
            reasoning_score=7,
        ),
        ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            priority=LLMPriority.COST,
            cost_per_1m_tokens=0.15,
            avg_latency_ms=600,
            reasoning_score=6,
        ),
        ModelConfig(
            name="qwen-turbo",
            provider="dashscope",
            priority=LLMPriority.COST,
            cost_per_1m_tokens=0.30,
            avg_latency_ms=500,
            reasoning_score=6,
        ),
    ],
    LLMPriority.QUALITY: [
        ModelConfig(
            name="claude-opus-4-20250514",
            provider="anthropic",
            priority=LLMPriority.QUALITY,
            cost_per_1m_tokens=15.0,
            avg_latency_ms=2000,
            reasoning_score=10,
        ),
        ModelConfig(
            name="claude-sonnet-4-20250514",
            provider="anthropic",
            priority=LLMPriority.QUALITY,
            cost_per_1m_tokens=3.0,
            avg_latency_ms=1200,
            reasoning_score=9,
        ),
        ModelConfig(
            name="gpt-4o",
            provider="openai",
            priority=LLMPriority.QUALITY,
            cost_per_1m_tokens=2.5,
            avg_latency_ms=1000,
            reasoning_score=8,
        ),
        ModelConfig(
            name="deepseek-reasoner",
            provider="deepseek",
            priority=LLMPriority.QUALITY,
            cost_per_1m_tokens=1.0,
            avg_latency_ms=5000,
            reasoning_score=9,
        ),
        ModelConfig(
            name="qwen-max",
            provider="dashscope",
            priority=LLMPriority.QUALITY,
            cost_per_1m_tokens=2.0,
            avg_latency_ms=1500,
            reasoning_score=8,
        ),
    ],
    LLMPriority.SPEED: [
        ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            priority=LLMPriority.SPEED,
            cost_per_1m_tokens=0.15,
            avg_latency_ms=400,
            reasoning_score=6,
        ),
        ModelConfig(
            name="claude-haiku-4-20250514",
            provider="anthropic",
            priority=LLMPriority.SPEED,
            cost_per_1m_tokens=0.25,
            avg_latency_ms=500,
            reasoning_score=6,
        ),
        ModelConfig(
            name="qwen-turbo",
            provider="dashscope",
            priority=LLMPriority.SPEED,
            cost_per_1m_tokens=0.30,
            avg_latency_ms=300,
            reasoning_score=6,
        ),
    ],
    LLMPriority.BALANCED: [
        ModelConfig(
            name="claude-sonnet-4-20250514",
            provider="anthropic",
            priority=LLMPriority.BALANCED,
            cost_per_1m_tokens=3.0,
            avg_latency_ms=1200,
            reasoning_score=9,
        ),
        ModelConfig(
            name="gpt-4o",
            provider="openai",
            priority=LLMPriority.BALANCED,
            cost_per_1m_tokens=2.5,
            avg_latency_ms=1000,
            reasoning_score=8,
        ),
        ModelConfig(
            name="qwen-plus",
            provider="dashscope",
            priority=LLMPriority.BALANCED,
            cost_per_1m_tokens=0.40,
            avg_latency_ms=800,
            reasoning_score=7,
        ),
    ],
}


class LLMProviderError(Exception):
    """LLM 提供商错误"""


class BaseLLMProvider(ABC):
    """LLM 提供商抽象基类"""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """创建模型实例"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查提供商是否可用"""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) 提供商"""

    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )
        except ImportError as e:
            raise LLMProviderError("langchain-anthropic 未安装") from e

    def is_available(self) -> bool:
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: F401

            return bool(self.api_key)
        except ImportError:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 提供商（兼容 API）"""

    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek 提供商"""

    def __init__(self) -> None:
        super().__init__(
            api_key=settings.llm_api_key or "",
            base_url=settings.llm_base_url or "https://api.deepseek.com",
        )

    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.api_key:
            raise LLMProviderError("DeepSeek API key 未配置")
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class DashScopeProvider(BaseLLMProvider):
    """阿里云 DashScope (Qwen) 提供商"""

    def __init__(self) -> None:
        super().__init__(
            api_key=settings.dashscope_api_key or settings.llm_api_key or "",
            base_url=settings.llm_base_url or settings.dashscope_base_url,
        )

    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.api_key:
            raise LLMProviderError("DashScope API key 未配置")
        return ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class OllamaProvider(BaseLLMProvider):
    """Ollama 本地提供商"""

    def __init__(self) -> None:
        super().__init__(
            api_key="",  # Ollama 不需要 API key
            base_url=settings.llm_base_url or "http://localhost:11434",
        )

    def create_model(self, model_name: str, **kwargs) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model_name,
                base_url=self.base_url,
                **kwargs,
            )
        except ImportError as e:
            raise LLMProviderError("langchain-ollama 未安装") from e

    def is_available(self) -> bool:
        try:
            from langchain_ollama import ChatOllama  # noqa: F401

            return True
        except ImportError:
            return False


# 提供商注册表
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "dashscope": DashScopeProvider,
    "ollama": OllamaProvider,
}


def get_provider(provider_name: str) -> BaseLLMProvider:
    """获取提供商实例"""
    provider_class = _PROVIDERS.get(provider_name)
    if provider_class is None:
        raise LLMProviderError(f"未知的提供商: {provider_name}")

    if provider_name in ("deepseek", "dashscope", "ollama"):
        return provider_class()

    api_key = settings.llm_api_key
    if not api_key:
        raise LLMProviderError(f"{provider_name} API key 未配置")

    return provider_class(api_key=api_key, base_url=settings.llm_base_url)


def get_llm_for_task(
    priority: LLMPriority = LLMPriority.BALANCED,
    fallback_priority: LLMPriority | None = None,
    **model_kwargs,
) -> BaseChatModel:
    """根据任务优先级获取 LLM"""
    configs = _MODEL_REGISTRY.get(priority, [])
    if not configs:
        raise LLMProviderError(f"未找到优先级为 {priority} 的模型配置")

    default_kwargs = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    default_kwargs.update(model_kwargs)

    # 尝试创建模型
    for config in configs:
        try:
            provider = get_provider(config.provider)
            if provider.is_available():
                llm = provider.create_model(config.name, **default_kwargs)
                logger.info(
                    "llm_selected",
                    model=config.name,
                    provider=config.provider,
                    priority=priority,
                    cost_per_1m=config.cost_per_1m_tokens,
                )
                return llm
        except LLMProviderError as e:
            logger.warning(
                "llm_provider_unavailable",
                provider=config.provider,
                error=str(e),
            )
            continue

    if fallback_priority and fallback_priority != priority:
        logger.info("fallback_to_secondary_priority", priority=fallback_priority)
        return get_llm_for_task(fallback_priority, **model_kwargs)

    raise LLMProviderError(f"无法创建优先级为 {priority} 的 LLM，请检查相关提供商的 API key 配置")


def get_model_configs(priority: LLMPriority | None = None) -> list[ModelConfig]:
    """获取模型配置列表"""
    if priority:
        return _MODEL_REGISTRY.get(priority, []).copy()

    all_configs = []
    for configs in _MODEL_REGISTRY.values():
        all_configs.extend(configs)
    return all_configs


def register_model_config(config: ModelConfig) -> None:
    """注册自定义模型配置"""
    if config.priority not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[config.priority] = []
    _MODEL_REGISTRY[config.priority].append(config)
    logger.info("model_config_registered", model=config.name, priority=config.priority)


__all__ = [
    "LLMPriority",
    "ModelConfig",
    "get_llm_for_task",
    "get_model_configs",
    "register_model_config",
    "get_provider",
    "LLMProviderError",
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "DashScopeProvider",
    "OllamaProvider",
]
