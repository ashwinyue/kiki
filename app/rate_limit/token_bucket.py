"""令牌桶限流中间件

基于令牌桶算法的限流实现，支持：
- 按键限流（IP、用户 ID 等）
- 可配置速率和容量
- 突发流量支持
- 优雅降级
- 标准响应头

参考: ai-engineer-training2/week09/3/p29限流中间件实现/限流中间件.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config.settings import get_settings
from app.observability.logging import bind_context, get_logger

settings = get_settings()
logger = get_logger(__name__)


class RateLimitExceeded(Exception):
    """限流超出异常"""

    def __init__(self, retry_after: float, limit: int, remaining: int):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f}s")


@dataclass
class TokenBucket:
    """令牌桶

    属性：
        rate: 令牌填充速率（令牌/秒）
        capacity: 桶容量（最大令牌数）
        tokens: 当前令牌数
        updated_at: 最后更新时间（单调时钟）
    """

    rate: float
    capacity: int
    tokens: float = field(init=False)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def refill(self) -> None:
        """根据经过的时间填充令牌，但不超过桶容量。"""
        now = time.monotonic()
        elapsed = now - self.updated_at

        if elapsed > 0:
            new_tokens = elapsed * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.updated_at = now

    async def consume(
        self,
        tokens: float = 1.0,
    ) -> tuple[bool, int, float]:
        """消费令牌

        Args:
            tokens: 要消费的令牌数

        Returns:
            tuple: (allowed, remaining, retry_after)
                - allowed: 是否允许消费
                - remaining: 剩余令牌数
                - retry_after: 需要等待的时间（秒）
        """
        self.refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            remaining = int(self.tokens)
            return True, remaining, 0.0

        needed = tokens - self.tokens
        retry_after = needed / self.rate if self.rate > 0 else float("inf")
        return False, 0, retry_after

    def time_to_full(self) -> float:
        """计算桶完全填满的时间

        Returns:
            float: 需要的秒数
        """
        self.refill()
        return (self.capacity - self.tokens) / self.rate if self.rate > 0 else 0.0


@dataclass
class RateLimitPolicy:
    """限流策略

    属性：
        rate: 每秒令牌数
        burst_capacity: 突发容量
        tokens_per_request: 每次请求消耗的令牌数
        key_func: 限流键提取函数
        exempt_paths: 豁免路径集合
        ttl_seconds: 桶过期时间（秒）
    """

    rate: float
    burst_capacity: int
    tokens_per_request: float = 1.0
    key_func: Callable[[Request], str] | None = None
    exempt_paths: set[str] = field(default_factory=set)
    ttl_seconds: int = 600

    CHAT = lambda cls: cls(rate=0.5, burst_capacity=10)
    CHAT_STREAM = lambda cls: cls(rate=0.33, burst_capacity=5)
    API = lambda cls: cls(rate=10.0, burst_capacity=50)
    HEALTH = lambda cls: cls(rate=20.0, burst_capacity=20)
    REGISTER = lambda cls: cls(rate=0.1, burst_capacity=5)
    LOGIN = lambda cls: cls(rate=0.33, burst_capacity=10)


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """令牌桶限流中间件

    基于 IP 地址或自定义键进行限流。

    特性：
    - 令牌桶算法，支持突发流量
    - 自动清理过期的桶
    - 标准 RateLimit 响应头
    - 可配置的豁免路径
    - 分布式友好的键提取

    使用示例：
        ```python
        from app.rate_limit.token_bucket import TokenBucketRateLimiter

        app.add_middleware(
            TokenBucketRateLimiter,
            rate_per_sec=10.0,
            burst_capacity=50,
            exempt_paths={"/health", "/metrics"},
        )
        ```
    """

    def __init__(
        self,
        app,
        rate_per_sec: float,
        burst_capacity: int,
        tokens_per_request: float = 1.0,
        key_func: Callable[[Request], str] | None = None,
        exempt_paths: set[str] | None = None,
        ttl_seconds: int = 600,
        max_buckets: int = 10000,
    ):
        """初始化限流中间件

        Args:
            app: ASGI 应用
            rate_per_sec: 令牌填充速率（令牌/秒）
            burst_capacity: 桶容量（最大令牌数）
            tokens_per_request: 每次请求消耗的令牌数
            key_func: 自定义限流键提取函数
            exempt_paths: 豁限限流的路径集合
            ttl_seconds: 桶过期时间（秒）
            max_buckets: 最大桶数量
        """
        super().__init__(app)

        self.rate = float(rate_per_sec)
        self.capacity = int(burst_capacity)
        self.tokens_per_request = float(tokens_per_request)
        self.key_func = key_func or self._default_key_func
        self.exempt_paths = exempt_paths or set()
        self.ttl_seconds = ttl_seconds
        self.max_buckets = max_buckets

        self._buckets: dict[str, TokenBucket] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = asyncio.Lock()

        self._policy_name = f"token_bucket; rate={self.rate}/s; burst={self.capacity}"

    def _default_key_func(self, request: Request) -> str:
        """限流键提取函数，优先级：X-User-ID > X-Forwarded-For > X-Real-IP > 客户端 IP。"""
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"

        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"

        xff = request.headers.get("X-Forwarded-For")
        if xff:
            ip = xff.split(",")[0].strip()
            return f"ip:{ip}"

        xri = request.headers.get("X-Real-IP")
        if xri:
            return f"ip:{xri}"

        if request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    async def _get_bucket(self, key: str) -> TokenBucket:
        """获取或创建令牌桶

        Args:
            key: 限流键

        Returns:
            TokenBucket: 令牌桶对象
        """
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    rate=self.rate,
                    capacity=self.capacity,
                )

            self._last_seen[key] = time.monotonic()
            return self._buckets[key]

    async def _cleanup_expired_buckets(self) -> None:
        """清理过期的令牌桶并防止内存泄漏。"""
        now = time.monotonic()
        cutoff = now - self.ttl_seconds

        async with self._lock:
            expired_keys = [
                k for k, last_seen in self._last_seen.items()
                if last_seen < cutoff
            ]

            for key in expired_keys:
                self._buckets.pop(key, None)
                self._last_seen.pop(key, None)

            if len(self._buckets) > self.max_buckets:
                sorted_keys = sorted(
                    self._last_seen.items(),
                    key=lambda x: x[1],
                )
                for key, _ in sorted_keys[:len(self._buckets) - self.max_buckets]:
                    self._buckets.pop(key, None)
                    self._last_seen.pop(key, None)

    def _create_rate_limit_response(
        self,
        retry_after: float,
        remaining: int,
    ) -> JSONResponse:
        """创建限流响应

        Args:
            retry_after: 重试时间（秒）
            remaining: 剩余令牌数

        Returns:
            JSONResponse: 429 响应
        """
        reset_time = max(0, int(retry_after))

        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "请求过于频繁，请稍后重试",
                "retry_after": reset_time,
            },
            headers={
                "Retry-After": str(reset_time),
                "X-RateLimit-Policy": self._policy_name,
                "X-RateLimit-Limit": str(self.capacity),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
            },
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """处理请求

        Args:
            request: FastAPI 请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP 响应
        """
        # 检查豁免路径
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # 获取限流键
        key = self.key_func(request)
        bind_context(rate_limit_key=key)

        # 定期清理过期桶
        if len(self._buckets) > 1000:
            asyncio.create_task(self._cleanup_expired_buckets())

        # 获取令牌桶并消费令牌
        bucket = await self._get_bucket(key)
        allowed, remaining, retry_after = await bucket.consume(
            self.tokens_per_request
        )

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                key=key,
                path=request.url.path,
                method=request.method,
                retry_after=retry_after,
            )

            return self._create_rate_limit_response(retry_after, remaining)

        # 请求正常，添加响应头
        response = await call_next(request)

        response.headers["X-RateLimit-Policy"] = self._policy_name
        response.headers["X-RateLimit-Limit"] = str(self.capacity)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(bucket.time_to_full()))

        return response


class PathBasedRateLimiter(TokenBucketRateLimiter):
    """基于路径的限流器

    不同路径可以使用不同的限流策略。
    """

    def __init__(
        self,
        app,
        policies: dict[str, RateLimitPolicy],
        default_policy: RateLimitPolicy | None = None,
        key_func: Callable[[Request], str] | None = None,
    ):
        """初始化基于路径的限流器

        Args:
            app: ASGI 应用
            policies: 路径到限流策略的映射
            default_policy: 默认限流策略
            key_func: 限流键提取函数
        """
        # 使用默认策略初始化父类
        default = default_policy or RateLimitPolicy(rate=10.0, burst_capacity=50)

        super().__init__(
            app,
            rate_per_sec=default.rate,
            burst_capacity=default.burst_capacity,
            tokens_per_request=default.tokens_per_request,
            key_func=key_func,
            exempt_paths=default.exempt_paths,
        )

        self.policies = policies

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """根据路径选择限流策略"""
        path = request.url.path

        # 查找匹配的策略
        policy = None
        for pattern, pol in self.policies.items():
            if path.startswith(pattern):
                policy = pol
                break

        if policy:
            # 临时替换限流参数
            original_rate = self.rate
            original_capacity = self.capacity
            original_tokens = self.tokens_per_request

            self.rate = policy.rate
            self.capacity = policy.burst_capacity
            self.tokens_per_request = policy.tokens_per_request

            try:
                return await super().dispatch(request, call_next)
            finally:
                # 恢复原始值
                self.rate = original_rate
                self.capacity = original_capacity
                self.tokens_per_request = original_tokens

        return await super().dispatch(request, call_next)


# ============== 便捷函数 ==============

def check_rate_limit(
    key: str,
    rate: float,
    capacity: int,
    tokens: float = 1.0,
) -> tuple[bool, float]:
    """检查限流（无状态方式）

    用于在非中间件场景下检查限流。

    Args:
        key: 限流键
        rate: 令牌填充速率
        capacity: 桶容量
        tokens: 需要消费的令牌数

    Returns:
        tuple: (allowed, retry_after)
    """
    # 简化实现：使用 Redis 存储桶状态
    # 这里只是一个示例，实际应用中需要持久化存储
    bucket = TokenBucket(rate=rate, capacity=capacity)
    allowed, remaining, retry_after = asyncio.run(bucket.consume(tokens))
    return allowed, retry_after


async def async_check_rate_limit(
    key: str,
    rate: float,
    capacity: int,
    tokens: float = 1.0,
) -> tuple[bool, float]:
    """异步检查限流

    Args:
        key: 限流键
        rate: 令牌填充速率
        capacity: 桶容量
        tokens: 需要消费的令牌数

    Returns:
        tuple: (allowed, retry_after)
    """
    bucket = TokenBucket(rate=rate, capacity=capacity)
    allowed, remaining, retry_after = await bucket.consume(tokens)
    return allowed, retry_after
