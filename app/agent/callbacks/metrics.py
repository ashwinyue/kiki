"""Prometheus 指标 Callback Handler

专门用于收集 Prometheus 指标的 Callback Handler。
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.observability.logging import get_logger

logger = get_logger(__name__)


class MetricsCallbackHandler(BaseCallbackHandler):
    """Prometheus 指标 Callback Handler

    专注于收集和记录 Prometheus 指标。
    与 KikiCallbackHandler 配合使用时，可以提供更详细的指标分解。

    Attributes:
        session_id: 会话 ID
        user_id: 用户 ID
        metrics_registry: Prometheus 指标注册表
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """初始化指标 Callback Handler

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            labels: 额外的标签
        """
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.labels = labels or {}

        # 统计信息
        self._llm_call_count = 0
        self._tool_call_count = 0
        self._total_tokens = 0

        # 延迟初始化 Prometheus 客户端
        self._metrics_initialized = False

        logger.debug(
            "metrics_callback_handler_initialized",
            session_id=session_id,
            user_id=user_id,
        )

    def _ensure_metrics(self) -> None:
        """确保 Prometheus 指标已初始化"""
        if self._metrics_initialized:
            return

        try:
            from prometheus_client import Counter, Histogram

            # 定义指标
            self.llm_duration = Histogram(
                "kiki_agent_llm_duration_seconds",
                "LLM 调用持续时间",
                ["model", "session_id"],
            )
            self.llm_tokens_total = Counter(
                "kiki_agent_llm_tokens_total",
                "LLM Token 总数",
                ["model", "token_type", "session_id"],
            )
            self.tool_duration = Histogram(
                "kiki_agent_tool_duration_seconds",
                "工具调用持续时间",
                ["tool_name", "session_id"],
            )
            self.tool_calls_total = Counter(
                "kiki_agent_tool_calls_total",
                "工具调用总数",
                ["tool_name", "status", "session_id"],
            )

            self._metrics_initialized = True
            logger.debug("prometheus_metrics_initialized")

        except ImportError:
            logger.warning("prometheus_client_not_installed")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时记录指标"""
        self._llm_call_count += 1

        try:
            self._ensure_metrics()

            # 提取模型名称
            model_name = "unknown"
            if response.llm_output:
                model_name = response.llm_output.get("model_name", "unknown")

            # 提取 token 使用
            token_usage = {}
            if hasattr(response.llm_output, "token_usage"):
                token_usage = response.llm_output.get("token_usage", {})

            # 记录 token 指标
            for token_type, count in token_usage.items():
                self.llm_tokens_total.labels(
                    model=model_name,
                    token_type=token_type,
                    session_id=self.session_id,
                ).inc(count)
                self._total_tokens += count

        except Exception as e:
            logger.debug("metrics_llm_record_failed", error=str(e))

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """工具调用结束时记录指标"""
        self._tool_call_count += 1

        try:
            self._ensure_metrics()

            tool_name = kwargs.get("name", "unknown")

            self.tool_calls_total.labels(
                tool_name=tool_name,
                status="success",
                session_id=self.session_id,
            ).inc()

        except Exception as e:
            logger.debug("metrics_tool_record_failed", error=str(e))

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """工具调用错误时记录指标"""
        try:
            self._ensure_metrics()

            tool_name = kwargs.get("name", "unknown")

            self.tool_calls_total.labels(
                tool_name=tool_name,
                status="error",
                session_id=self.session_id,
            ).inc()

        except Exception as e:
            logger.debug("metrics_tool_error_record_failed", error=str(e))

    def get_stats(self) -> dict[str, int]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "llm_call_count": self._llm_call_count,
            "tool_call_count": self._tool_call_count,
            "total_tokens": self._total_tokens,
        }
