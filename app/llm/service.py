"""LLM 服务

提供 LLM 调用、重试和循环回退容错。
"""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import BaseModel

from app.config.settings import get_settings
from app.llm.registry import LLMRegistry
from app.observability.logging import get_logger
from app.utils.retry_decorators import (
    RetryableError,
    create_llm_retry_decorator,
)

try:
    from app.llm.providers import (
        LLMPriority,
        LLMProviderError,
    )
    from app.llm.providers import (
        get_llm_for_task as get_llm_for_task_providers,
    )

    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False

logger = get_logger(__name__)

settings = get_settings()

T = TypeVar("T", bound=BaseModel)


def resolve_provider(model_name: str | None) -> str:
    """根据模型名称推断提供商"""
    if not model_name:
        return settings.llm_provider

    name = model_name.lower()
    if name.startswith("gpt-") or name.startswith("o1-") or name.startswith("o3-"):
        return "openai"
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("qwen-"):
        return "dashscope"
    if name.startswith("deepseek-"):
        return "deepseek"
    if name.startswith("llama") or name.startswith("mistral") or name.startswith("phi"):
        return "ollama"

    return settings.llm_provider


class LLMService:
    """LLM 服务"""

    def __init__(
        self,
        default_model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        """初始化 LLM 服务"""
        self._default_model = default_model or settings.llm_model
        self._max_retries = max_retries
        self._current_model_index = 0
        self._llm: BaseChatModel | None = None
        self._raw_llm: BaseChatModel | None = None

        self._init_current_model()

        logger.info(
            "llm_service_initialized",
            default_model=self._default_model,
            max_retries=max_retries,
        )

    def _init_current_model(self) -> None:
        """初始化当前 LLM"""
        try:
            self._raw_llm = LLMRegistry.get(self._default_model)
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
                self._raw_llm = LLMRegistry.get(self._default_model)
                self._llm = self._get_llm_with_retry(self._default_model)
                logger.info("using_first_available_model", model=self._default_model)

    def _get_llm_with_retry(self, model_name: str) -> BaseChatModel:
        """获取带重试配置的 LLM"""
        llm = LLMRegistry.get(model_name)
        return self._apply_retry(llm)

    def get_raw_llm(self) -> BaseChatModel | None:
        """获取原始 LLM 实例（不带重试包装）"""
        return self._raw_llm

    def get_llm_with_tools(self, tools: list[Any] | None = None) -> BaseChatModel | None:
        """获取带工具绑定和重试配置的 LLM"""
        if self._raw_llm is None:
            return None

        llm = self._raw_llm
        if tools:
            llm = llm.bind_tools(tools)

        return self._apply_retry(llm)

    def _switch_to_next_model(self) -> bool:
        """切换到下一个模型（循环）"""
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
            self._raw_llm = LLMRegistry.get(next_model)
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
        """调用 LLM"""
        if model_name:
            try:
                self._raw_llm = LLMRegistry.get(model_name)
                llm = self._get_llm_with_retry(model_name)
                models = LLMRegistry.list_models()
                if model_name in models:
                    self._current_model_index = models.index(model_name)
                response = await llm.ainvoke(messages)
                return response
            except ValueError:
                logger.error("requested_model_not_found", model=model_name)
                raise

        models = LLMRegistry.list_models()
        models_tried = 0
        last_error: Exception | None = None

        while models_tried < len(models):
            try:
                if not self._llm:
                    self._raw_llm = LLMRegistry.get(self._default_model)
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

        raise RuntimeError(f"LLM 调用失败，已尝试 {models_tried} 个模型。最后错误: {last_error}")

    async def call_with_tenacity_retry(
        self,
        messages: list[BaseMessage],
        model_name: str | None = None,
        max_attempts: int = 3,
        **kwargs,
    ) -> BaseMessage:
        """使用 tenacity 装饰器的 LLM 调用方法"""
        @create_llm_retry_decorator(max_attempts=max_attempts)
        async def _do_call() -> BaseMessage:
            if not self._llm:
                raise RuntimeError("LLM 未初始化")
            return await self._llm.ainvoke(messages, **kwargs)

        try:
            return await _do_call()
        except Exception as e:
            if isinstance(e, RateLimitError):
                raise RetryableError(
                    f"API 速率限制: {e}",
                    retry_after=5.0,
                ) from e
            if isinstance(e, APITimeoutError):
                raise RetryableError(
                    f"API 超时: {e}",
                    retry_after=2.0,
                ) from e
            raise

    def bind_tools(self, tools: list[Any]) -> "LLMService":
        """绑定工具到当前 LLM"""
        if self._raw_llm:
            self._raw_llm = self._raw_llm.bind_tools(tools)
            self._llm = self._apply_retry(self._raw_llm)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    def _apply_retry(self, llm: BaseChatModel) -> BaseChatModel:
        """为 LLM 应用重试配置"""
        return llm.with_retry(
            stop_after_attempt=self._max_retries,
            retry_if_exception_type=(RateLimitError, APITimeoutError, APIError),
        )

    def with_structured_output(self, schema: type[T]) -> BaseChatModel:
        """获取带结构化输出的 LLM"""
        llm = self._raw_llm or LLMRegistry.get(self._default_model)
        structured_llm = llm.with_structured_output(schema)
        return self._apply_retry(structured_llm)

    @property
    def current_model(self) -> str:
        """获取当前模型名称"""
        return self._default_model

    def get_llm(self) -> BaseChatModel | None:
        """获取当前 LLM 实例"""
        return self._llm

    def get_llm_with_retry(self) -> BaseChatModel | None:
        """获取带重试配置的 LLM 实例"""
        if self._llm is None:
            if self._raw_llm is None:
                self._raw_llm = LLMRegistry.get(self._default_model)
            self._llm = self._apply_retry(self._raw_llm)
        return self._llm

    def get_llm_for_task(
        self,
        priority: str = "balanced",
        **model_kwargs,
    ) -> BaseChatModel:
        """根据任务优先级获取 LLM（多模型路由）"""
        if not _PROVIDERS_AVAILABLE:
            logger.warning("llm_providers_not_available", fallback="default_model")
            llm = self.get_llm_with_retry()
            if llm is None:
                raise RuntimeError("LLM 未初始化")
            return llm

        try:
            priority_enum = LLMPriority(priority)
        except ValueError:
            logger.warning("invalid_priority", priority=priority, fallback="balanced")
            priority_enum = LLMPriority.BALANCED

        try:
            llm = get_llm_for_task_providers(priority=priority_enum, **model_kwargs)
            logger.info(
                "llm_for_task_selected",
                priority=priority,
                model=llm.model_name if hasattr(llm, "model_name") else "unknown",
            )
            return llm.with_retry(
                stop_after_attempt=self._max_retries,
                retry_if_exception_type=(RateLimitError, APITimeoutError, APIError),
            )
        except LLMProviderError as e:
            logger.error("llm_provider_error", error=str(e))
            llm = self.get_llm_with_retry()
            if llm is None:
                raise RuntimeError(f"无法创建 LLM: {e}") from e
            return llm

    async def chat(
        self,
        messages: list[dict[str, str] | BaseMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """聊天接口（兼容 ChatPipeline）"""
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                lc_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

        invoke_kwargs: dict[str, Any] = {
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        if config:
            invoke_kwargs["config"] = config

        llm = self.get_llm_with_retry()
        if model:
            llm = LLMRegistry.get(model)
            llm = self._apply_retry(llm)

        if llm is None:
            raise RuntimeError("LLM 未初始化")

        response: BaseMessage = await llm.ainvoke(lc_messages, **invoke_kwargs)

        return {"content": str(response.content), "response": response}

    async def chat_stream(
        self,
        messages: list[dict[str, str] | BaseMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式聊天接口（兼容 ChatPipeline）"""
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                lc_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

        invoke_kwargs: dict[str, Any] = {
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        if config:
            invoke_kwargs["config"] = config

        llm = self.get_llm_with_retry()
        if model:
            llm = LLMRegistry.get(model)
            llm = self._apply_retry(llm)

        if llm is None:
            raise RuntimeError("LLM 未初始化")

        async for chunk in llm.astream(lc_messages, **invoke_kwargs):
            if hasattr(chunk, "content") and chunk.content:
                yield str(chunk.content)


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """获取全局 LLM 服务实例（单例）"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
