"""LLM 模型注册表"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LLMRegistry:
    """LLM 模型注册表"""

    _models: dict[str, dict[str, Any]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """确保模型已初始化"""
        if not cls._initialized:
            _init_default_models()
            cls._initialized = True

    @classmethod
    def register(
        cls,
        name: str,
        llm: BaseChatModel,
        description: str | None = None,
    ) -> None:
        """注册 LLM 模型"""
        cls._models[name] = {
            "name": name,
            "llm": llm,
            "description": description,
        }
        logger.info("llm_registered", model_name=name)

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseChatModel:
        """获取 LLM 模型"""
        cls._ensure_initialized()

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"LLM 模型 '{name}' 未注册。可用模型: {available}")

        if kwargs:
            logger.debug(
                "creating_llm_with_custom_args", model_name=name, kwargs=list(kwargs.keys())
            )
            base_llm = cls._models[name]["llm"]
            if isinstance(base_llm, ChatOpenAI):
                if name.startswith("qwen-"):
                    api_key = settings.dashscope_api_key or settings.llm_api_key
                    base_url = settings.llm_base_url or settings.dashscope_base_url
                    return ChatOpenAI(
                        model=name,
                        api_key=api_key,
                        base_url=base_url,
                        **kwargs,
                    )
                return ChatOpenAI(
                    model=name,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    **kwargs,
                )
            return base_llm

        logger.debug("using_registered_llm", model_name=name)
        return cls._models[name]["llm"]

    @classmethod
    def list_models(cls) -> list[str]:
        """列出所有已注册的模型名称"""
        cls._ensure_initialized()
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查模型是否已注册"""
        cls._ensure_initialized()
        return name in cls._models


def _init_default_models() -> None:
    """初始化默认的 LLM 模型"""
    if settings.llm_provider == "openai":
        models = [
            ("gpt-4o", "GPT-4O - 高性能多模态模型"),
            ("gpt-4o-mini", "GPT-4O Mini - 快速轻量级模型"),
            ("gpt-3.5-turbo", "GPT-3.5 Turbo - 经济型模型"),
        ]
        for model_name, description in models:
            try:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                LLMRegistry.register(model_name, llm, description)
            except Exception as e:
                logger.warning("failed_to_register_llm", model_name=model_name, error=str(e))

    elif settings.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            models = [
                ("claude-sonnet-4-20250514", "Claude Sonnet 4 - 平衡性能与速度"),
                ("claude-opus-4-20250514", "Claude Opus 4 - 最强推理能力"),
                ("claude-haiku-4-20250514", "Claude Haiku 4 - 快速响应"),
            ]
            for model_name, description in models:
                try:
                    llm = ChatAnthropic(
                        model=model_name,
                        api_key=settings.llm_api_key,
                        temperature=settings.llm_temperature,
                        max_tokens=settings.llm_max_tokens,
                    )
                    LLMRegistry.register(model_name, llm, description)
                except Exception as e:
                    logger.warning("failed_to_register_llm", model_name=model_name, error=str(e))
        except ImportError:
            logger.warning("langchain_anthropic_not_installed")

    elif settings.llm_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            # Ollama 默认模型
            llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.llm_base_url or "http://localhost:11434",
                temperature=settings.llm_temperature,
            )
            LLMRegistry.register(settings.llm_model, llm, "Ollama 本地模型")
        except ImportError:
            logger.warning("langchain_ollama_not_installed")

    elif settings.llm_provider == "deepseek":
        api_key = settings.deepseek_api_key or settings.llm_api_key
        if not api_key:
            logger.warning("deepseek_api_key_not_configured")
            return

        base_url = settings.llm_base_url or settings.deepseek_base_url

        deepseek_models = [
            ("deepseek-chat", "DeepSeek Chat - 高性价比通用模型"),
            ("deepseek-reasoner", "DeepSeek Reasoner - 强推理能力模型"),
        ]

        for model_name, description in deepseek_models:
            try:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                LLMRegistry.register(model_name, llm, description)
                logger.info("deepseek_model_registered", model_name=model_name)
            except Exception as e:
                logger.warning("failed_to_register_deepseek", model_name=model_name, error=str(e))

    elif settings.llm_provider == "dashscope":
        api_key = settings.dashscope_api_key or settings.llm_api_key
        if not api_key:
            logger.warning("dashscope_api_key_not_configured")
            return

        base_url = settings.llm_base_url or settings.dashscope_base_url

        qwen_models = [
            ("qwen-max", "Qwen Max - 最强推理能力，适合复杂任务"),
            ("qwen-plus", "Qwen Plus - 平衡性能与成本"),
            ("qwen-turbo", "Qwen Turbo - 快速响应，适合简单任务"),
            ("qwen-long", "Qwen Long - 长上下文支持（最高 1M tokens）"),
        ]

        for model_name, description in qwen_models:
            try:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                LLMRegistry.register(model_name, llm, description)
                logger.info("qwen_model_registered", model_name=model_name)
            except Exception as e:
                logger.warning("failed_to_register_qwen", model_name=model_name, error=str(e))
