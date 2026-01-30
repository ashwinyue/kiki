"""LLM 服务

提供 LLM 模型注册表、使用 LangChain 内置重试机制、结构化输出支持。
"""

from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

# 泛型类型用于结构化输出
T = TypeVar("T", bound=BaseModel)


class LLMRegistry:
    """LLM 模型注册表

    维护可用的 LLM 配置列表，按名称检索。
    """

    _models: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        llm: BaseChatModel,
        description: str | None = None,
    ) -> None:
        """注册 LLM 模型

        Args:
            name: 模型名称
            llm: LLM 实例
            description: 模型描述
        """
        cls._models[name] = {
            "name": name,
            "llm": llm,
            "description": description,
        }
        logger.info("llm_registered", model_name=name)

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseChatModel:
        """获取 LLM 模型

        Args:
            name: 模型名称
            **kwargs: 覆盖默认配置的参数

        Returns:
            BaseChatModel 实例

        Raises:
            ValueError: 如果模型名称未注册
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"LLM 模型 '{name}' 未注册。可用模型: {available}"
            )

        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=name, kwargs=list(kwargs.keys()))
            # 根据模型类型创建新实例
            base_llm = cls._models[name]["llm"]
            if isinstance(base_llm, ChatOpenAI):
                return ChatOpenAI(
                    model=name,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    **kwargs,
                )
            # 其他 LLM 类型可以在这里添加
            return base_llm

        logger.debug("using_registered_llm", model_name=name)
        return cls._models[name]["llm"]

    @classmethod
    def list_models(cls) -> list[str]:
        """列出所有已注册的模型名称

        Returns:
            模型名称列表
        """
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查模型是否已注册

        Args:
            name: 模型名称

        Returns:
            是否已注册
        """
        return name in cls._models


def _init_default_models() -> None:
    """初始化默认的 LLM 模型"""

    # OpenAI 模型
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

    # Anthropic 模型
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

    # Ollama 模型
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


# 初始化默认模型
_init_default_models()


class LLMService:
    """LLM 服务

    提供 LLM 调用、重试和循环回退容错。
    使用 LangChain 内置的 with_retry 方法替代手动重试。
    """

    def __init__(
        self,
        default_model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """初始化 LLM 服务

        Args:
            default_model: 默认模型名称
            max_retries: 最大重试次数
        """
        self._default_model = default_model or settings.llm_model
        self._max_retries = max_retries
        self._current_model_index = 0
        self._llm: BaseChatModel | None = None

        # 初始化当前模型
        self._init_current_model()

        logger.info(
            "llm_service_initialized",
            default_model=self._default_model,
            max_retries=max_retries,
        )

    def _init_current_model(self) -> None:
        """初始化当前 LLM"""
        try:
            self._llm = self._get_llm_with_retry(self._default_model)
            models = LLMRegistry.list_models()
            if self._default_model in models:
                self._current_model_index = models.index(self._default_model)
        except ValueError as e:
            logger.warning("default_model_not_found", model=self._default_model, error=str(e))
            models = LLMRegistry.list_models()
            if models:
                self._default_model = models[0]
                self._current_model_index = 0
                self._llm = self._get_llm_with_retry(self._default_model)
                logger.info("using_first_available_model", model=self._default_model)

    def _get_llm_with_retry(self, model_name: str) -> BaseChatModel:
        """获取带重试配置的 LLM

        使用 LangChain 的 with_retry 方法配置重试策略。

        Args:
            model_name: 模型名称

        Returns:
            配置了重试的 LLM 实例
        """
        llm = LLMRegistry.get(model_name)

        # 使用 LangChain 的 with_retry 配置重试
        # 参考: https://python.langchain.com/docs/how_to/retry/
        return llm.with_retry(
            stop_after_attempt=self._max_retries,
            # 针对 OpenAI 错误的重试配置
            retry_on_exception=lambda e: isinstance(e, (RateLimitError, APITimeoutError, APIError)),
        )

    def _switch_to_next_model(self) -> bool:
        """切换到下一个模型（循环）

        Returns:
            是否成功切换
        """
        models = LLMRegistry.list_models()
        if len(models) <= 1:
            return False

        next_index = (self._current_model_index + 1) % len(models)
        next_model = models[next_index]

        logger.warning(
            "switching_to_next_model",
            from_model=self._default_model,
            to_model=next_model,
        )

        try:
            self._llm = self._get_llm_with_retry(next_model)
            self._default_model = next_model
            self._current_model_index = next_index
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
            return False

    async def call(
        self,
        messages: list[BaseMessage],
        model_name: str | None = None,
        **kwargs,
    ) -> BaseMessage:
        """调用 LLM

        Args:
            messages: 消息列表
            model_name: 指定模型名称（可选）
            **kwargs: 模型参数覆盖

        Returns:
            LLM 响应消息

        Raises:
            RuntimeError: 所有模型调用失败
        """
        # 如果指定了模型，切换到该模型
        if model_name:
            try:
                llm = self._get_llm_with_retry(model_name)
                models = LLMRegistry.list_models()
                if model_name in models:
                    self._current_model_index = models.index(model_name)
                response = await llm.ainvoke(messages)
                return response
            except ValueError:
                logger.error("requested_model_not_found", model=model_name)
                raise

        # 尝试所有可用模型
        models = LLMRegistry.list_models()
        models_tried = 0
        last_error: Exception | None = None

        while models_tried < len(models):
            try:
                if not self._llm:
                    self._llm = self._get_llm_with_retry(self._default_model)

                response = await self._llm.ainvoke(messages)
                return response
            except OpenAIError as e:
                last_error = e
                models_tried += 1
                logger.error(
                    "llm_call_failed",
                    model=self._default_model,
                    tried=models_tried,
                    total=len(models),
                    error=str(e),
                )

                # 切换到下一个模型
                if models_tried < len(models) and self._switch_to_next_model():
                    continue
                break

        # 所有模型都失败了
        raise RuntimeError(
            f"LLM 调用失败，已尝试 {models_tried} 个模型。"
            f"最后错误: {last_error}"
        )

    def bind_tools(self, tools: list[Any]) -> "LLMService":
        """绑定工具到当前 LLM

        Args:
            tools: 工具列表

        Returns:
            self，支持链式调用
        """
        if self._llm:
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    def with_structured_output(self, schema: type[T]) -> BaseChatModel:
        """获取带结构化输出的 LLM

        使用 LangChain 的 with_structured_output 方法
        确保返回符合 Pydantic 模型的结构化数据。

        Args:
            schema: Pydantic 模型类

        Returns:
            配置了结构化输出的 LLM

        Examples:
            ```python
            from pydantic import BaseModel, Field

            class RouteDecision(BaseModel):
                agent: str = Field(description="目标 agent 名称")
                reason: str = Field(description="选择原因")

            structured_llm = llm_service.with_structured_output(RouteDecision)
            decision: RouteDecision = await structured_llm.ainvoke(messages)
            ```
        """
        llm = self._llm or self._get_llm_with_retry(self._default_model)

        # 使用 LangChain 的 with_structured_output
        # 参考: https://python.langchain.com/docs/how_to/structured_output/
        return llm.with_structured_output(schema)

    @property
    def current_model(self) -> str:
        """获取当前模型名称"""
        return self._default_model

    def get_llm(self) -> BaseChatModel | None:
        """获取当前 LLM 实例

        Returns:
            当前 LLM 实例
        """
        return self._llm

    def get_llm_with_retry(self) -> BaseChatModel | None:
        """获取带重试配置的 LLM 实例

        确保返回的 LLM 已配置重试策略。

        Returns:
            配置了重试的 LLM 实例
        """
        if self._llm is None:
            self._llm = self._get_llm_with_retry(self._default_model)
        return self._llm


# 全局 LLM 服务实例
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """获取全局 LLM 服务实例（单例）

    Returns:
        LLMService 实例
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
