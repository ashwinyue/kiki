"""中间件

包含：
- ObservabilityMiddleware: 统一可观测性中间件（日志、指标）
- RequestContextMiddleware: 请求上下文中间件
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import get_logger, bind_context, clear_context


logger = get_logger(__name__)


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
        user_id = request.state.get("user_id") or request.headers.get("X-User-ID")

        if user_id:
            bind_context(user_id=user_id)

        # 提取会话 ID
        session_id = request.state.get("session_id") or request.headers.get("X-Session-ID")
        if session_id:
            bind_context(session_id=session_id)

        response = await call_next(request)
        return response
