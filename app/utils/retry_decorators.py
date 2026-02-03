"""
Tenacity 重试装饰器模块

提供生产级重试策略，与现有 app/retry 模块互补。
使用 tenacity 库实现灵活的重试机制。

参考：aold/ai-engineer-training2/projects/project1_2/utils/retry.py
"""

from typing import Any, Callable, TypeVar
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.config import settings

logger = logging.getLogger(__name__)

# 泛型类型 - 简化定义避免类型推断问题
T = TypeVar("T")
F = TypeVar("F")


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: int = 2,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    创建自定义重试装饰器

    Args:
        max_attempts: 最大重试次数（默认：3）
        min_wait: 最小等待时间（秒）（默认：1.0）
        max_wait: 最大等待时间（秒）（默认：10.0）
        exponential_base: 指数退避基数（默认：2）
        exceptions: 需要重试的异常类型元组

    Returns:
        重试装饰器函数

    Example:
        ```python
        @create_retry_decorator(max_attempts=5, min_wait=2.0)
        async def fetch_data(url: str) -> dict:
            async with aiohttp.get(url) as resp:
                return await resp.json()
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=min_wait,
            max=max_wait,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def create_llm_retry_decorator(
    max_attempts: int | None = None,
    min_wait: float | None = None,
    max_wait: float | None = None,
) -> Callable[[F], F]:
    """
    创建 LLM 服务专用重试装饰器

    使用配置文件中的设置，提供默认值。

    Args:
        max_attempts: 最大重试次数（默认从配置读取）
        min_wait: 最小等待时间（默认从配置读取）
        max_wait: 最大等待时间（默认从配置读取）

    Returns:
        LLM 专用的重试装饰器

    Example:
        ```python
        @create_llm_retry_decorator()
        async def call_llm(messages: list[Message]) -> Message:
            return await llm_service.generate(messages)
        ```
    """
    # 从配置获取默认值
    _max_attempts = max_attempts or getattr(settings, "LLM_MAX_RETRIES", 3)
    _min_wait = min_wait or getattr(settings, "LLM_RETRY_MIN_WAIT", 1.0)
    _max_wait = max_wait or getattr(settings, "LLM_RETRY_MAX_WAIT", 10.0)

    return create_retry_decorator(
        max_attempts=_max_attempts,
        min_wait=_min_wait,
        max_wait=_max_wait,
        exceptions=(
            ConnectionError,
            TimeoutError,
            # 可以添加更多 LLM 相关异常
        ),
    )


def retry_with_backoff(
    func: F | None = None,
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> F:
    """
    带退避的重试装饰器（简化版）

    可以直接使用或带参数使用。

    Args:
        func: 被装饰的函数
        max_attempts: 最大重试次数
        min_wait: 最小等待时间
        max_wait: 最大等待时间

    Example:
        ```python
        # 无参数使用
        @retry_with_backoff
        def risky_operation():
            pass

        # 带参数使用
        @retry_with_backoff(max_attempts=5, min_wait=2.0)
        def another_operation():
            pass
        ```
    """
    def decorator(f: F) -> F:
        return create_retry_decorator(
            max_attempts=max_attempts,
            min_wait=min_wait,
            max_wait=max_wait,
        )(f)  # type: ignore[return-value]

    if func is None:
        # 带参数调用：@retry_with_backoff(...)
        return decorator  # type: ignore[return-value]
    else:
        # 无参数调用：@retry_with_backoff
        return decorator(func)


class RetryableError(Exception):
    """可重试的错误基类"""

    def __init__(self, message: str, *, retry_after: float | None = None):
        """
        Args:
            message: 错误消息
            retry_after: 建议重试时间（秒），None 表示立即重试
        """
        super().__init__(message)
        self.retry_after = retry_after


class NonRetryableError(Exception):
    """不可重试的错误（将立即抛出）"""
    pass


def is_retryable_error(exception: Exception) -> bool:
    """
    判断异常是否可重试

    Args:
        exception: 异常对象

    Returns:
        如果可重试返回 True
    """
    if isinstance(exception, NonRetryableError):
        return False
    if isinstance(exception, RetryableError):
        return True
    # 默认不可重试
    return False


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    **kwargs: Any,
) -> Any:
    """
    异步函数重试包装器

    Args:
        func: 异步函数
        *args: 位置参数
        max_attempts: 最大重试次数
        min_wait: 最小等待时间
        max_wait: 最大等待时间
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Raises:
        最后一次调用的异常

    Example:
        ```python
        result = await retry_async(
            fetch_data,
            "https://api.example.com",
            max_attempts=5,
        )
        ```
    """
    import asyncio

    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == max_attempts:
                # 最后一次尝试失败
                logger.error(
                    f"函数 {func.__name__} 在 {max_attempts} 次尝试后失败"
                )
                raise

            # 计算等待时间（指数退避）
            wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)

            # 检查是否是 RetryableError 且有指定的重试时间
            if isinstance(e, RetryableError) and e.retry_after is not None:
                wait_time = e.retry_after

            logger.warning(
                f"函数 {func.__name__} 第 {attempt} 次尝试失败: {e}. "
                f"等待 {wait_time:.1f} 秒后重试..."
            )
            await asyncio.sleep(wait_time)

    # 理论上不会到达这里，但为了类型检查
    assert last_exception is not None
    raise last_exception


def retry_sync(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    **kwargs: Any,
) -> Any:
    """
    同步函数重试包装器

    Args:
        func: 同步函数
        *args: 位置参数
        max_attempts: 最大重试次数
        min_wait: 最小等待时间
        max_wait: 最大等待时间
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Raises:
        最后一次调用的异常
    """
    import time

    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(
                    f"函数 {func.__name__} 在 {max_attempts} 次尝试后失败"
                )
                raise

            wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)

            if isinstance(e, RetryableError) and e.retry_after is not None:
                wait_time = e.retry_after

            logger.warning(
                f"函数 {func.__name__} 第 {attempt} 次尝试失败: {e}. "
                f"等待 {wait_time:.1f} 秒后重试..."
            )
            time.sleep(wait_time)

    assert last_exception is not None
    raise last_exception
