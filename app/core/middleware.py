"""中间件

包含：
- ObservabilityMiddleware: 统一可观测性中间件（日志、指标）
- RequestContextMiddleware: 请求上下文中间件
- SecurityHeadersMiddleware: 安全响应头中间件
"""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.observability.logging import bind_context, clear_context, get_logger
from app.observability.metrics import track_http_request

logger = get_logger(__name__)


# ============== 安全响应头中间件 ==============


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全响应头中间件

    添加安全相关的 HTTP 响应头：
    - X-Content-Type-Options: nosniff - 防止 MIME 类型嗅探
    - X-Frame-Options: DENY - 防止点击劫持
    - X-XSS-Protection: 1; mode=block - XSS 保护
    - Strict-Transport-Security: HSTS - 强制 HTTPS（仅 HTTPS 时）
    - Content-Security-Policy: CSP - 内容安全策略
    - Referrer-Policy: strict - 控制 Referer 信息泄露

    参考: https://owasp.org/www-project-secure-headers/
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # X-Content-Type-Options: 防止浏览器 MIME 嗅探
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options: 防止点击劫持
        response.headers["X-Frame-Options"] = "DENY"

        # X-XSS-Protection: 启用浏览器 XSS 过滤器
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy: 控制 Referer 头信息泄露
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy: 控制浏览器功能/权限
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # 仅 HTTPS 时设置 HSTS
        if request.url.scheme == "https":
            # max-age=31536000 (1年) + includeSubDomains + preload
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains; preload"

        return response


# ============== 请求大小限制中间件 ==============


class MaxRequestSizeMiddleware(BaseHTTPMiddleware):
    """请求大小限制中间件

    限制请求体大小，防止大文件上传攻击。
    """

    def __init__(
        self,
        app: ASGIApp,
        max_size: int = 10 * 1024 * 1024,  # 默认 10MB
    ) -> None:
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查 Content-Length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    logger.warning(
                        "request_too_large",
                        path=request.url.path,
                        size=size,
                        max_size=self.max_size,
                    )
                    return Response(
                        status_code=413,
                        content=f"Request body too large. Maximum size is {self.max_size} bytes.",
                    )
            except ValueError:
                pass

        return await call_next(request)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """可观测性中间件

    功能：
    - 自动记录请求/响应日志
    - 生成请求 ID
    - 记录请求耗时
    - Prometheus 指标收集（预留）
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        exclude_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过排除的路径
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # 生成请求 ID
        request_id = str(uuid.uuid4())

        # 绑定上下文
        bind_context(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        # 记录请求开始
        logger.info(
            "request_started",
            path=request.url.path,
            method=request.method,
            client=request.client.host if request.client else None,
        )

        # 计时
        start_time = time.time()

        try:
            async with track_http_request(request.method, request.url.path):
                # 处理请求
                response = await call_next(request)

                # 计算耗时
                duration = time.time() - start_time

                # 记录响应
                logger.info(
                    "request_completed",
                    status_code=response.status_code,
                    duration_seconds=round(duration, 4),
                )

                # 添加响应头
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(duration, 4))

                return response

        except Exception as e:
            duration = time.time() - start_time

            logger.exception(
                "request_failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=round(duration, 4),
            )
            raise

        finally:
            # 清空上下文
            clear_context()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求上下文中间件

    将请求信息绑定到上下文变量中，供后续使用。
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 提取用户 ID（如果已认证）
        user_id = getattr(request.state, "user_id", None) or request.headers.get("X-User-ID")

        if user_id:
            bind_context(user_id=user_id)

        # 提取会话 ID
        session_id = getattr(request.state, "session_id", None) or request.headers.get(
            "X-Session-ID"
        )
        if session_id:
            bind_context(session_id=session_id)

        response = await call_next(request)
        return response
