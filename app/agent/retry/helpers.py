"""重试工具函数模块

提供重试相关的工具函数。

使用示例:
```python
from app.agent.retry.helpers import execute_with_retry, create_retryable_node

# 执行带重试的函数
result = await execute_with_retry(
    llm.ainvoke,
    messages,
    policy=RetryPolicy(max_attempts=3)
)

# 创建可重试的节点
retryable_node = create_retryable_node(
    my_node,
    policy=RetryPolicy(max_attempts=3)
)
```
"""

import asyncio
from collections.abc import Callable
from typing import Any

from app.agent.retry.strategy import RetryPolicy, get_default_retry_policy
from app.observability.logging import get_logger

logger = get_logger(__name__)


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
    "execute_with_retry",
    "create_retryable_node",
]
