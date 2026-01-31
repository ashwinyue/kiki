"""工具重试（Tool Retry）模块

基于 LangGraph 的 retry 参数实现工具调用的自动重试机制。

核心特性：
- 可配置的重试策略（最大次数、重试间隔、退避因子）
- 支持指定可重试的异常类型
- 指数退避算法避免雪崩
- 支持自定义重试条件

参考实现: week07/p13-toolRetry.py
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Type

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RetryStrategy(str, Enum):
    """重试策略枚举"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"  # 指数退避
    LINEAR_BACKOFF = "linear_backoff"  # 线性退避
    FIXED_INTERVAL = "fixed_interval"  # 固定间隔
    IMMEDIATE = "immediate"  # 立即重试


# ========== 自定义异常类型 ==========


class RetryableError(Exception):
    """可重试错误基类

    所有继承此类的异常都会被自动重试。
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """初始化可重试错误

        Args:
            message: 错误消息
            retry_after: 建议重试等待时间（秒）
        """
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(RetryableError):
    """网络错误（超时、连接失败等）"""

    pass


class RateLimitError(RetryableError):
    """速率限制错误"""

    def __init__(
        self,
        message: str = "请求过于频繁，已被限流",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, retry_after or 5.0)


class ResourceUnavailableError(RetryableError):
    """资源不可用错误（如数据库连接池满）"""

    pass


class TemporaryServiceError(RetryableError):
    """临时服务错误（服务暂时不可用）"""

    pass


class ToolExecutionError(Exception):
    """工具执行错误（默认不重试）"""

    pass


# ========== 重试策略配置 ==========


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
    retry_on: tuple[Type[Exception], ...] = (
        NetworkError,
        RateLimitError,
        ResourceUnavailableError,
        TemporaryServiceError,
    )
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
        # 检查最大尝试次数
        if attempt >= self.max_attempts:
            return False

        # 检查自定义条件
        if self.custom_condition is not None:
            try:
                return self.custom_condition(exception, attempt)
            except Exception as e:
                logger.warning(
                    "custom_condition_failed",
                    error=str(e),
                )
                return False

        # 检查是否重试所有异常
        if self.retry_on_any:
            return True

        # 检查异常类型
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

        # 应用最大间隔限制
        delay = min(delay, self.max_interval)

        # 应用抖动
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


# ========== 默认重试策略 ==========


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


# ========== 重试装饰器 ==========


def with_retry(
    policy: RetryPolicy | None = None,
    on_retry: Callable[[Exception, int], Any] | None = None,
) -> Callable:
    """为函数添加重试能力的装饰器

    Args:
        policy: 重试策略（None 则使用默认策略）
        on_retry: 重试时的回调函数

    Returns:
        装饰后的函数

    Examples:
        ```python
        @with_retry(max_attempts=3)
        async def risky_operation():
            # 可能失败的操作
            pass
        ```
    """
    if policy is None:
        policy = get_default_retry_policy()

    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            last_exception = None

            while attempt <= policy.max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e, attempt):
                        logger.error(
                            "retry_exhausted_non_retryable",
                            function=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        raise

                    delay = policy.get_retry_delay(attempt)
                    logger.info(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                    # 调用回调
                    if on_retry:
                        try:
                            await on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.warning(
                                "on_retry_callback_failed",
                                error=str(callback_error),
                            )

                    # 等待后重试
                    if delay > 0:
                        await asyncio.sleep(delay)

                    attempt += 1

            # 所有重试都失败
            logger.error(
                "retry_exhausted",
                function=func.__name__,
                attempts=attempt - 1,
                error=str(last_exception),
            )
            raise last_exception

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            last_exception = None

            while attempt <= policy.max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e, attempt):
                        logger.error(
                            "retry_exhausted_non_retryable",
                            function=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        raise

                    delay = policy.get_retry_delay(attempt)
                    logger.info(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                    # 调用回调
                    if on_retry:
                        try:
                            if asyncio.iscoroutinefunction(on_retry):
                                asyncio.run(on_retry(e, attempt))
                            else:
                                on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.warning(
                                "on_retry_callback_failed",
                                error=str(callback_error),
                            )

                    # 等待后重试
                    if delay > 0:
                        time.sleep(delay)

                    attempt += 1

            # 所有重试都失败
            logger.error(
                "retry_exhausted",
                function=func.__name__,
                attempts=attempt - 1,
                error=str(last_exception),
            )
            raise last_exception

        # 根据函数类型返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ========== 重试上下文管理器 ==========


class RetryContext:
    """重试上下文管理器

    用于在代码块中执行可重试的操作。

    Examples:
        ```python
        policy = RetryPolicy(max_attempts=3)

        async with RetryContext(policy) as retry:
            await retry.attempt(risky_operation)
        ```
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        """初始化重试上下文

        Args:
            policy: 重试策略
        """
        self.policy = policy or get_default_retry_policy()
        self._attempt = 1

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # 不抑制异常

    async def attempt(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """执行带重试的函数调用

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值

        Raises:
            最后一次失败的异常
        """
        last_exception = None

        while self._attempt <= self.policy.max_attempts:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.policy.should_retry(e, self._attempt):
                    raise

                delay = self.policy.get_retry_delay(self._attempt)
                logger.info(
                    "retry_context_attempt",
                    attempt=self._attempt,
                    max_attempts=self.policy.max_attempts,
                    delay=delay,
                    error=str(e),
                )

                if delay > 0:
                    await asyncio.sleep(delay)

                self._attempt += 1

        raise last_exception


# ========== 工具执行包装器 ==========


async def execute_with_retry(
    func: Callable,
    *args: Any,
    policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> Any:
    """执行函数并应用重试策略

    便捷函数，用于一次性执行带重试的函数调用。

    Args:
        func: 要执行的函数
        *args: 位置参数
        policy: 重试策略
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Examples:
        ```python
        result = await execute_with_retry(
            llm.ainvoke,
            messages,
            policy=RetryPolicy(max_attempts=3)
        )
        ```
    """
    if policy is None:
        policy = get_default_retry_policy()

    attempt = 1
    last_exception = None

    while attempt <= policy.max_attempts:
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not policy.should_retry(e, attempt):
                raise

            delay = policy.get_retry_delay(attempt)
            logger.info(
                "execute_with_retry_attempt",
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay=delay,
                error=str(e),
            )

            if delay > 0:
                await asyncio.sleep(delay)

            attempt += 1

    raise last_exception


# ========== LangGraph 节点重试辅助函数 ==========


def create_retryable_node(
    node_func: Callable,
    policy: RetryPolicy | None = None,
) -> Callable:
    """创建可重试的节点函数

    包装 LangGraph 节点函数，使其具有自动重试能力。

    Args:
        node_func: 原始节点函数
        policy: 重试策略

    Returns:
        包装后的节点函数

    Examples:
        ```python
        async def my_node(state: AgentState) -> dict:
            # 节点逻辑
            return {"messages": [...]}

        retryable_node = create_retryable_node(
            my_node,
            policy=RetryPolicy(max_attempts=3)
        )

        builder.add_node("my_node", retryable_node)
        ```
    """
    if policy is None:
        policy = get_default_retry_policy()

    async def wrapped_node(state, config=None) -> dict:
        attempt = 1
        last_exception = None

        while attempt <= policy.max_attempts:
            try:
                return await node_func(state, config)
            except Exception as e:
                last_exception = e

                if not policy.should_retry(e, attempt):
                    logger.error(
                        "node_retry_failed_non_retryable",
                        node=node_func.__name__,
                        attempt=attempt,
                        error=str(e),
                    )
                    # 返回错误状态而不是抛出异常
                    return {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "retry_attempts": attempt - 1,
                    }

                delay = policy.get_retry_delay(attempt)
                logger.info(
                    "node_retry_attempt",
                    node=node_func.__name__,
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    delay=delay,
                    error=str(e),
                )

                if delay > 0:
                    await asyncio.sleep(delay)

                attempt += 1

        # 所有重试失败
        logger.error(
            "node_retry_exhausted",
            node=node_func.__name__,
            attempts=attempt - 1,
            error=str(last_exception),
        )
        return {
            "error": str(last_exception),
            "error_type": type(last_exception).__name__,
            "retry_attempts": attempt - 1,
        }

    # 保留原始函数的元数据
    wrapped_node.__name__ = node_func.__name__
    wrapped_node.__doc__ = node_func.__doc__
    wrapped_node.__module__ = node_func.__module__

    return wrapped_node


__all__ = [
    # 异常类型
    "RetryableError",
    "NetworkError",
    "RateLimitError",
    "ResourceUnavailableError",
    "TemporaryServiceError",
    "ToolExecutionError",
    # 策略相关
    "RetryStrategy",
    "RetryPolicy",
    "get_default_retry_policy",
    # 装饰器
    "with_retry",
    # 上下文管理器
    "RetryContext",
    # 工具函数
    "execute_with_retry",
    "create_retryable_node",
]
