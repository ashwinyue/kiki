"""重试装饰器模块

提供为函数添加重试能力的装饰器。

使用示例:
```python
from app.agent.retry.decorator import with_retry

@with_retry(max_attempts=3)
async def risky_operation():
    # 可能失败的操作
    pass
```
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from app.agent.retry.strategy import RetryPolicy, get_default_retry_policy
from app.observability.logging import get_logger

logger = get_logger(__name__)


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


__all__ = [
    "with_retry",
]
