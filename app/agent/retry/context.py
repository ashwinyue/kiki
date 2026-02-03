"""重试上下文管理器模块

提供用于执行可重试操作的上下文管理器。
"""

import asyncio
from collections.abc import Callable
from typing import Any

from app.agent.retry.strategy import RetryPolicy, get_default_retry_policy
from app.observability.logging import get_logger

logger = get_logger(__name__)


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


__all__ = [
    "RetryContext",
]
