"""统一的 Kiki Callback Handler

集成 Langfuse 追踪、Prometheus 指标和结构化日志。
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from app.config.settings import get_settings
from app.llm import resolve_provider
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class KikiCallbackHandler(BaseCallbackHandler):
    """统一的 Kiki Callback Handler

    提供：
    - LLM 调用的完整生命周期追踪
    - 工具调用的记录
    - Token 使用统计
    - 错误追踪

    Attributes:
        session_id: 会话 ID
        user_id: 用户 ID
        enable_langfuse: 是否启用 Langfuse 追踪
        enable_metrics: 是否启用 Prometheus 指标
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        enable_langfuse: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """初始化 Callback Handler

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            enable_langfuse: 是否启用 Langfuse 追踪
            enable_metrics: 是否启用 Prometheus 指标
        """
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.enable_langfuse = enable_langfuse
        self.enable_metrics = enable_metrics

        # 统计信息
        self._llm_start_time: float | None = None
        self._tool_start_time: float | None = None
        self._token_usage: dict[str, int] = {}

        # Langfuse 客户端（延迟初始化）
        self._langfuse = None

        logger.debug(
            "callback_handler_initialized",
            session_id=session_id,
            user_id=user_id,
        )

    @property
    def langfuse(self):
        """获取 Langfuse 客户端（延迟初始化）"""
        if self._langfuse is None and self.enable_langfuse and settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except ImportError:
                logger.warning("langfuse_not_installed")
            except Exception as e:
                logger.warning("langfuse_init_failed", error=str(e))

        return self._langfuse

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始时的回调

        Args:
            serialized: 序列化的 LLM 信息
            prompts: 提示词列表
            **kwargs: 额外参数
        """
        import time

        self._llm_start_time = time.time()

        logger.info(
            "llm_start",
            session_id=self.session_id,
            user_id=self.user_id,
            model=serialized.get("name", "unknown"),
            prompt_count=len(prompts),
        )

        # Langfuse 追踪
        if self.langfuse:
            try:
                self.langfuse.score(
                    name="llm_start",
                    value=1.0,
                    comment=f"LLM调用开始: {serialized.get('name')}",
                )
            except Exception as e:
                logger.debug("langfuse_score_failed", error=str(e))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时的回调

        Args:
            response: LLM 响应结果
            **kwargs: 额外参数
        """
        import time

        duration = None
        if self._llm_start_time:
            duration = time.time() - self._llm_start_time
            self._llm_start_time = None

        # 提取 token 使用情况
        token_usage = {}
        if response.llm_output and hasattr(response.llm_output, "token_usage"):
            token_usage = response.llm_output.get("token_usage", {})
            self._token_usage = token_usage

        logger.info(
            "llm_end",
            session_id=self.session_id,
            user_id=self.user_id,
            duration_seconds=duration,
            token_usage=token_usage,
            generations_count=len(response.generations),
        )

        # Prometheus 指标
        if self.enable_metrics:
            try:
                from app.observability import metrics

                model_name = response.llm_output.get("model_name", "unknown")
                provider = response.llm_output.get("provider") or resolve_provider(model_name)
                if isinstance(response.llm_output, dict):
                    response.llm_output.setdefault("provider", provider)

                if duration:
                    metrics.llm_duration_seconds.labels(
                        model=model_name,
                        provider=provider,
                    ).observe(duration)

                metrics.llm_requests_total.labels(
                    model=model_name,
                    provider=provider,
                    status="success",
                ).inc()

                prompt_tokens = token_usage.get("prompt_tokens")
                completion_tokens = token_usage.get("completion_tokens")
                if isinstance(prompt_tokens, int):
                    metrics.llm_tokens_total.labels(
                        model=model_name,
                        token_type="prompt",
                    ).inc(prompt_tokens)
                if isinstance(completion_tokens, int):
                    metrics.llm_tokens_total.labels(
                        model=model_name,
                        token_type="completion",
                    ).inc(completion_tokens)
            except ImportError:
                pass
            except Exception as e:
                logger.debug("metrics_record_failed", error=str(e))

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """LLM 调用出错时的回调

        Args:
            error: 异常对象
            **kwargs: 额外参数
        """
        self._llm_start_time = None

        logger.error(
            "llm_error",
            session_id=self.session_id,
            user_id=self.user_id,
            error_type=type(error).__name__,
            error_message=str(error),
        )

        # Prometheus 错误指标
        if self.enable_metrics:
            try:
                from app.observability import metrics

                metrics.llm_requests_total.labels(
                    model="unknown",
                    provider=resolve_provider(None),
                    status="error",
                ).inc()
            except ImportError:
                pass
            except Exception as e:
                logger.debug("metrics_error_record_failed", error=str(e))

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """聊天模型开始时的回调

        Args:
            serialized: 序列化的模型信息
            messages: 消息列表
            **kwargs: 额外参数
        """
        import time

        self._llm_start_time = time.time()

        logger.debug(
            "chat_model_start",
            session_id=self.session_id,
            user_id=self.user_id,
            model=serialized.get("name", "unknown"),
            message_count=len(messages[0]) if messages else 0,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """工具调用开始时的回调

        Args:
            serialized: 序列化的工具信息
            input_str: 工具输入
            **kwargs: 额外参数
        """
        import time

        self._tool_start_time = time.time()

        logger.info(
            "tool_start",
            session_id=self.session_id,
            user_id=self.user_id,
            tool_name=serialized.get("name", "unknown"),
            input_length=len(input_str),
        )

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """工具调用结束时的回调

        Args:
            output: 工具输出
            **kwargs: 额外参数
        """
        import time

        duration = None
        if self._tool_start_time:
            duration = time.time() - self._tool_start_time
            self._tool_start_time = None

        logger.info(
            "tool_end",
            session_id=self.session_id,
            user_id=self.user_id,
            duration_seconds=duration,
            output_length=len(output) if output else 0,
        )

        # Prometheus 工具指标
        if self.enable_metrics and duration:
            try:
                from app.observability import metrics

                metrics.tool_duration_seconds.labels(
                    tool_name=kwargs.get("name", "unknown"),
                ).observe(duration)
            except (ImportError, AttributeError):
                pass
            except Exception as e:
                logger.debug("tool_metrics_failed", error=str(e))

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """工具调用出错时的回调

        Args:
            error: 异常对象
            **kwargs: 额外参数
        """
        self._tool_start_time = None

        logger.error(
            "tool_error",
            session_id=self.session_id,
            user_id=self.user_id,
            tool_name=kwargs.get("name", "unknown"),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def get_token_usage(self) -> dict[str, int]:
        """获取 token 使用统计

        Returns:
            Token 使用统计字典
        """
        return self._token_usage.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._token_usage = {}
        self._llm_start_time = None
        self._tool_start_time = None
