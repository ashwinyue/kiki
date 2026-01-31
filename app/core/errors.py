"""错误处理模块

提供细粒度的错误分类和处理策略。
"""

from collections.abc import Callable
from enum import Enum

from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from app.core.config import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ErrorSeverity(str, Enum):
    """错误严重程度"""

    LOW = "low"  # 可恢复，不影响主流程
    MEDIUM = "medium"  # 需要重试或降级
    HIGH = "high"  # 需要立即处理
    CRITICAL = "critical"  # 系统级错误


class ErrorCategory(str, Enum):
    """错误类别"""

    RATE_LIMIT = "rate_limit"  # 速率限制
    TIMEOUT = "timeout"  # 超时
    CONNECTION = "connection"  # 连接错误
    VALIDATION = "validation"  # 验证错误
    AUTHENTICATION = "authentication"  # 认证错误
    PERMISSION = "permission"  # 权限错误
    TOOL_EXECUTION = "tool_execution"  # 工具执行错误
    LLM_ERROR = "llm_error"  # LLM 调用错误
    UNKNOWN = "unknown"  # 未知错误


class ErrorContext:
    """错误上下文

    包含错误的所有相关信息，用于处理策略决策。
    """

    def __init__(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        is_retryable: bool = False,
        user_message: str | None = None,
    ) -> None:
        self.error = error
        self.category = category
        self.severity = severity
        self.is_retryable = is_retryable
        self.user_message = user_message
        self.error_type = type(error).__name__
        self.error_message = str(error)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "error_type": self.error_type,
            "is_retryable": self.is_retryable,
        }


def classify_error(error: Exception) -> ErrorContext:
    """分类错误

    根据错误类型和内容，确定错误类别、严重程度和处理策略。

    Args:
        error: 异常对象

    Returns:
        ErrorContext: 错误上下文
    """
    # 速率限制错误
    if isinstance(error, RateLimitError):
        return ErrorContext(
            error=error,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            user_message="请求过于频繁，请稍后重试",
        )

    # 超时错误
    if isinstance(error, APITimeoutError):
        return ErrorContext(
            error=error,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            user_message="请求超时，请稍后重试",
        )

    # 连接错误
    if isinstance(error, APIConnectionError):
        return ErrorContext(
            error=error,
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            is_retryable=True,
            user_message="网络连接失败，请检查网络后重试",
        )

    # 验证错误
    if isinstance(error, ValidationError):
        return ErrorContext(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=False,
            user_message="输入数据格式不正确",
        )

    # 检查错误消息中的关键词
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # 认证错误
    if "auth" in error_msg or "unauthorized" in error_msg or error_type == "AuthenticationError":
        return ErrorContext(
            error=error,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            is_retryable=False,
            user_message="认证失败，请检查密钥配置",
        )

    # 权限错误
    if "permission" in error_msg or "forbidden" in error_msg:
        return ErrorContext(
            error=error,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            is_retryable=False,
            user_message="权限不足",
        )

    # 工具执行错误
    if "tool" in error_msg or error_type.endswith("ToolError"):
        return ErrorContext(
            error=error,
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            user_message="工具执行失败，请稍后重试",
        )

    # 默认未知错误
    return ErrorContext(
        error=error,
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.HIGH,
        is_retryable=False,
        user_message="发生未知错误，请联系管理员",
    )


def get_user_friendly_message(error: Exception, context: ErrorContext | None = None) -> str:
    """获取用户友好的错误消息

    生产环境不暴露敏感信息，开发环境显示详细信息。

    Args:
        error: 异常对象
        context: 错误上下文（可选）

    Returns:
        用户友好的错误消息
    """
    if context is None:
        context = classify_error(error)

    # 生产环境使用预定义消息
    if settings.is_production:
        return context.user_message or "操作失败，请稍后重试"

    # 开发环境显示详细信息
    return f"{context.user_message or '操作失败'}: [{context.error_type}] {context.error_message}"


def handle_tool_error(error: Exception) -> str:
    """处理工具执行错误

    根据环境返回适当的错误消息。

    Args:
        error: 工具执行异常

    Returns:
        错误消息
    """
    context = classify_error(error)

    logger.warning(
        "tool_execution_failed",
        **context.to_dict(),
    )

    return get_user_friendly_message(error, context)


class RetryStrategy:
    """重试策略

    根据错误类别决定是否重试以及重试间隔。
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试

        Args:
            error: 异常对象
            attempt: 当前尝试次数

        Returns:
            是否应该重试
        """
        if attempt >= self.max_retries:
            return False

        context = classify_error(error)
        return context.is_retryable

    def get_delay(self, attempt: int) -> float:
        """获取重试延迟

        Args:
            attempt: 当前尝试次数

        Returns:
            延迟时间（秒）
        """
        delay = self.initial_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> any:
        """执行函数并在失败时重试

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            Exception: 重试耗尽后抛出原始异常
        """
        import asyncio

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if not self.should_retry(e, attempt):
                    logger.error(
                        "execution_failed_non_retryable",
                        attempt=attempt,
                        error_type=type(e).__name__,
                    )
                    raise

                delay = self.get_delay(attempt)
                context = classify_error(e)

                logger.warning(
                    "execution_failed_retrying",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay=delay,
                    category=context.category.value,
                )

                await asyncio.sleep(delay)

        # 重试耗尽
        logger.error(
            "execution_failed_retry_exhausted",
            attempts=self.max_retries + 1,
            error_type=type(last_error).__name__,
        )
        raise last_error


# 默认重试策略
default_retry_strategy = RetryStrategy()
