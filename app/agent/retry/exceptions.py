"""重试异常类型定义模块

定义所有与重试相关的异常类型。

使用示例:
```python
from app.agent.retry.exceptions import (
    RetryableError,
    NetworkError,
    RateLimitError,
)

# 抛出可重试异常
raise NetworkError("连接超时", retry_after=2.0)
```
"""


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


__all__ = [
    "RetryableError",
    "NetworkError",
    "RateLimitError",
    "ResourceUnavailableError",
    "TemporaryServiceError",
    "ToolExecutionError",
]
