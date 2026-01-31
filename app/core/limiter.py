"""限流器模块

使用 slowapi 实现端点级别限流。
"""

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


def _get_identifier(request) -> str:
    """获取限流标识符

    优先使用用户 ID，其次使用 IP 地址。

    Args:
        request: FastAPI 请求对象

    Returns:
        限流标识符
    """
    # 尝试从请求状态获取用户 ID
    if hasattr(request.state, "user_id") and request.state.user_id:
        return f"user:{request.state.user_id}"

    # 尝试从请求头获取用户 ID
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"

    # 使用 IP 地址
    return get_remote_address(request)


# 初始化限流器
limiter = Limiter(
    key_func=_get_identifier,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=settings.redis_url,  # 使用 Redis 实现分布式限流
    enabled=settings.is_development or settings.is_production,
)


async def rate_limit_exceeded_handler(request, exc: RateLimitExceeded):
    """限流超出异常处理

    Args:
        request: FastAPI 请求对象
        exc: 限流异常

    Returns:
        JSON 响应
    """
    logger.warning(
        "rate_limit_exceeded",
        path=request.url.path,
        limit=exc.description,
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"请求过于频繁，请稍后再试。限制: {exc.description}",
            "retry_after": 60,  # 建议重试时间（秒）
        },
    )


# 便捷装饰器
def limit(
    limit_value: str,
    per_method: bool = True,
):
    """限流装饰器

    Args:
        limit_value: 限流值，如 "10 per minute"
        per_method: 是否按 HTTP 方法分别限流

    Returns:
        装饰器函数

    Examples:
        ```python
        @router.get("/search")
        @limit("10 per minute")
        async def search(query: str):
            ...
        ```
    """
    return limiter.limit(limit_value, per_method=per_method)


# 预定义的限流规则
class RateLimit:
    """预定义的限流规则"""

    CHAT = ["30 per minute", "500 per day"]
    CHAT_STREAM = ["20 per minute", "300 per day"]
    REGISTER = ["10 per hour", "50 per day"]
    LOGIN = ["20 per minute", "100 per day"]
    API = ["100 per minute", "1000 per day"]
    HEALTH = ["20 per minute"]
