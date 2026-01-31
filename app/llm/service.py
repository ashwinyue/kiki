"""LLM 服务

提供 LLM 调用、重试和循环回退容错。
使用 LangChain 内置的 with_retry 方法替代手动重试。
"""

from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import BaseModel

from app.config.settings import get_settings
from app.llm.registry import LLMRegistry
from app.observability.logging import get_logger

# 尝试导入多模型路由系统（可选依赖）
try:
    from app.llm_providers import (
        LLMPriority,
        LLMProviderError,
    )
    from app.llm_providers import (
        get_llm_for_task as get_llm_for_task_providers,
    )

    _PROVIDERS_AVAILABLE = True
except ImportError:
    _PROVIDERS_AVAILABLE = False

logger = get_logger(__name__)

settings = get_settings()

# 泛型类型用于结构化输出
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
    """LLM 服务

    提供 LLM 调用、重试和循环回退容错。
    使用 LangChain 内置的 with_retry 方法替代手动重试。

    重要：bind_tools 等配置方法必须在 with_retry 之前调用，
    因为 RunnableRetry 不支持 bind_tools 等方法。
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
        self._raw_llm: BaseChatModel | None = None  # 原始 LLM（不带重试）

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
        """获取带重试配置的 LLM

        使用 LangChain 的 with_retry 方法配置重试策略。

        Args:
            model_name: 模型名称

        Returns:
            配置了重试的 LLM 实例
        """
        llm = LLMRegistry.get(model_name)
        return self._apply_retry(llm)

    def get_raw_llm(self) -> BaseChatModel | None:
        """获取原始 LLM 实例（不带重试包装）

        用于需要绑定工具或其他配置的场景。
        绑定工具后，可以通过 get_llm_with_retry() 获取带重试的版本。

        Returns:
            原始 LLM 实例

        Examples:
            ```python
            # 正确的绑定工具方式
            raw_llm = llm_service.get_raw_llm()
            if raw_llm and tools:
                llm_with_tools = raw_llm.bind_tools(tools)
            ```
        """
        return self._raw_llm

    def get_llm_with_tools(self, tools: list[Any] | None = None) -> BaseChatModel | None:
        """获取带工具绑定和重试配置的 LLM

        这是获取 LLM 用于 LangGraph 的推荐方法。

        Args:
            tools: 工具列表（可选）

        Returns:
            配置好的 LLM 实例

        Examples:
            ```python
            # 获取带工具的 LLM
            llm = llm_service.get_llm_with_tools(tools)
            response = await llm.ainvoke(messages)
            ```
        """
        if self._raw_llm is None:
            return None

        llm = self._raw_llm
        if tools:
            llm = llm.bind_tools(tools)

        return self._apply_retry(llm)

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

        # 尝试所有可用模型
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

        # 所有模型都失败了
        raise RuntimeError(f"LLM 调用失败，已尝试 {models_tried} 个模型。最后错误: {last_error}")

    def bind_tools(self, tools: list[Any]) -> "LLMService":
        """绑定工具到当前 LLM

        注意：必须在重试包装之前绑定工具，因为 RunnableRetry 不支持 bind_tools。

        Args:
            tools: 工具列表

        Returns:
            self，支持链式调用
        """
        if self._raw_llm:
            # 在原始 LLM 上绑定工具
            self._raw_llm = self._raw_llm.bind_tools(tools)
            # 重新应用重试包装
            self._llm = self._apply_retry(self._raw_llm)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self

    def _apply_retry(self, llm: BaseChatModel) -> BaseChatModel:
        """为 LLM 应用重试配置

        Args:
            llm: 原始 LLM 实例

        Returns:
            配置了重试的 LLM 实例
        """
        # 使用 LangChain 的 with_retry 配置重试
        # 参考: https://python.langchain.com/docs/how_to/retry/
        return llm.with_retry(
            stop_after_attempt=self._max_retries,
            # 针对 OpenAI 错误的重试配置
            retry_if_exception_type=(RateLimitError, APITimeoutError, APIError),
        )

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
        llm = self._raw_llm or LLMRegistry.get(self._default_model)

        # 使用 LangChain 的 with_structured_output
        # 参考: https://python.langchain.com/docs/how_to/structured_output/
        structured_llm = llm.with_structured_output(schema)

        # 应用重试配置
        return self._apply_retry(structured_llm)

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
            if self._raw_llm is None:
                self._raw_llm = LLMRegistry.get(self._default_model)
            self._llm = self._apply_retry(self._raw_llm)
        return self._llm

    def get_llm_for_task(
        self,
        priority: str = "balanced",
        **model_kwargs,
    ) -> BaseChatModel:
        """根据任务优先级获取 LLM（多模型路由）

        这是 langgraph-agents 推荐的成本优化方法。
        根据任务类型自动选择最合适的模型：
        - cost: 简单任务，使用便宜模型（DeepSeek/Qwen-Turbo）
        - quality: 复杂任务，使用强模型（Claude Opus/Sonnet）
        - speed: 需要快速响应（GPT-4o-mini/Haiku）
        - balanced: 平衡性能与成本（Sonnet/GPT-4o）

        Args:
            priority: 任务优先级 (cost/quality/speed/balanced)
            **model_kwargs: 模型参数覆盖

        Returns:
            配置好的 LLM 实例

        Raises:
            RuntimeError: 如果多模型路由不可用或创建失败

        Examples:
            ```python
            # 简单任务用便宜模型
            llm_cheap = llm_service.get_llm_for_task("cost")

            # 复杂任务用强模型
            llm_smart = llm_service.get_llm_for_task("quality")
            ```
        """
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
            # 应用重试配置
            return llm.with_retry(
                stop_after_attempt=self._max_retries,
                retry_if_exception_type=(RateLimitError, APITimeoutError, APIError),
            )
        except LLMProviderError as e:
            logger.error("llm_provider_error", error=str(e))
            # 回退到默认模型
            llm = self.get_llm_with_retry()
            if llm is None:
                raise RuntimeError(f"无法创建 LLM: {e}") from e
            return llm


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
