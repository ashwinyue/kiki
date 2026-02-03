"""安全中间件

包含：
- SecurityHeadersMiddleware: 安全响应头中间件
- MaxRequestSizeMiddleware: 请求大小限制中间件
"""

from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.observability.logging import get_logger

logger = get_logger(__name__)


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

        response.headers["X-Content-Type-Options"] = "nosniff"

        response.headers["X-Frame-Options"] = "DENY"

        response.headers["X-XSS-Protection"] = "1; mode=block"

        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        if request.url.scheme == "https":
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains; preload"

        return response


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
