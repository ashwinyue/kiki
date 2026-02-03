"""工具重试（Tool Retry）模块

基于 LangGraph 的 retry 参数实现工具调用的自动重试机制。

核心特性：
- 可配置的重试策略（最大次数、重试间隔、退避因子）
- 支持指定可重试的异常类型
- 指数退避算法避免雪崩
- 支持自定义重试条件

模块化结构：
- exceptions: 异常类型定义
- strategy: 重试策略和配置
- decorator: 重试装饰器
- context: 重试上下文管理器
- helpers: 工具函数

使用示例:
```python
from app.agent.retry import (
    # 异常类型
    RetryableError,
    NetworkError,
    RateLimitError,
    # 策略相关
    RetryStrategy,
    RetryPolicy,
    get_default_retry_policy,
    # 装饰器
    with_retry,
    # 上下文管理器
    RetryContext,
    # 工具函数
    execute_with_retry,
    create_retryable_node,
)

# 使用装饰器
@with_retry(max_attempts=3)
async def risky_operation():
    pass

# 使用上下文管理器
async with RetryContext(policy) as retry:
    await retry.attempt(risky_operation)

# 执行带重试的函数
result = await execute_with_retry(func, arg1, arg2)

# 创建可重试的节点
retryable_node = create_retryable_node(my_node, policy)
```
"""

# ============== 异常类型 ==============
# ============== 上下文管理器 ==============
from app.agent.retry.context import (
    RetryContext,
)

# ============== 装饰器 ==============
from app.agent.retry.decorator import (
    with_retry,
)
from app.agent.retry.exceptions import (
    NetworkError,
    RateLimitError,
    ResourceUnavailableError,
    RetryableError,
    TemporaryServiceError,
    ToolExecutionError,
)

# ============== 工具函数 ==============
from app.agent.retry.helpers import (
    create_retryable_node,
    execute_with_retry,
)

# ============== 策略相关 ==============
from app.agent.retry.strategy import (
    RetryPolicy,
    RetryStrategy,
    get_default_retry_policy,
)

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
