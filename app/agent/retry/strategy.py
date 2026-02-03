"""重试策略配置模块

定义重试策略枚举和配置类。
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RetryStrategy(str, Enum):
    """重试策略枚举"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE = "immediate"


@dataclass
class RetryPolicy:
    """重试策略配置类

    定义工具节点的重试行为。

    Attributes:
        max_attempts: 最大重试次数（包括首次尝试）
        retry_on: 可重试的异常类型元组
        retry_on_any: 是否重试所有异常（危险，仅用于测试）
        strategy: 重试策略
        initial_interval: 初始重试间隔（秒）
        backoff_factor: 退避因子（指数退避时使用）
        max_interval: 最大重试间隔（秒）
        jitter: 是否添加随机抖动（避免惊群效应）
        jitter_percent: 抖动百分比（0.0-1.0）
        custom_condition: 自定义重试条件函数
    """

    max_attempts: int = 3
    retry_on: tuple[type[Exception], ...] = (
        # 延迟导入避免循环依赖
        lambda: (
            __import__("app.agent.retry.exceptions", fromlist=["NetworkError"]).NetworkError,
            __import__("app.agent.retry.exceptions", fromlist=["RateLimitError"]).RateLimitError,
            __import__(
                "app.agent.retry.exceptions", fromlist=["ResourceUnavailableError"]
            ).ResourceUnavailableError,
            __import__(
                "app.agent.retry.exceptions", fromlist=["TemporaryServiceError"]
            ).TemporaryServiceError,
        )
    )()
    retry_on_any: bool = False
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = True
    jitter_percent: float = 0.1
    custom_condition: Callable[[Exception, int], bool] | None = None

    # 内部计数（运行时使用）
    _attempt_count: int = field(default=0, init=False, repr=False)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试

        Args:
            exception: 发生的异常
            attempt: 当前尝试次数（从 1 开始）

        Returns:
            是否应该重试
        """
        if attempt >= self.max_attempts:
            return False

        if self.custom_condition is not None:
            try:
                return self.custom_condition(exception, attempt)
            except Exception as e:
                logger.warning(
                    "custom_condition_failed",
                    error=str(e),
                )
                return False

        if self.retry_on_any:
            return True

        for retryable_type in self.retry_on:
            if isinstance(exception, retryable_type):
                return True

        return False

    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟时间

        Args:
            attempt: 当前尝试次数（从 1 开始）

        Returns:
            延迟时间（秒）
        """
        if self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif self.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.initial_interval
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_interval * attempt
        else:  # EXPONENTIAL_BACKOFF
            delay = self.initial_interval * (self.backoff_factor ** (attempt - 1))

        delay = min(delay, self.max_interval)

        if self.jitter and delay > 0:
            jitter_range = delay * self.jitter_percent
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    def to_langgraph_retry(self) -> dict:
        """转换为 LangGraph retry 参数格式

        LangGraph 使用 dataclass 作为 retry 参数，
        这个方法返回一个兼容的配置字典。

        Returns:
            retry 配置字典
        """
        return {
            "max_attempts": self.max_attempts,
            "retry_on": self.retry_on,
            "initial_interval": self.initial_interval,
            "backoff_factor": self.backoff_factor,
        }


def get_default_retry_policy() -> RetryPolicy:
    """获取默认重试策略

    从配置文件读取默认值。

    Returns:
        RetryPolicy 实例
    """
    settings = get_settings()

    return RetryPolicy(
        max_attempts=settings.agent_max_retries,
        initial_interval=settings.agent_retry_initial_interval,
        backoff_factor=settings.agent_retry_backoff_factor,
    )


__all__ = [
    "RetryStrategy",
    "RetryPolicy",
    "get_default_retry_policy",
]
