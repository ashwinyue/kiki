"""应用入口

创建 FastAPI 应用并配置所有组件。
"""

import asyncio
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.gzip import GZipMiddleware

# 加载 .env 文件（必须在导入 app 模块之前执行）
load_dotenv()
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from app.config.errors import classify_error, get_user_friendly_message
from app.config.settings import get_settings
from app.middleware import (
    MaxRequestSizeMiddleware,
    ObservabilityMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    TenantMiddleware,
)
from app.observability.logging import configure_logging, get_logger
from app.rate_limit.limiter import RateLimit, limiter, rate_limit_exceeded_handler

# 获取配置
settings = get_settings()

# 配置日志
configure_logging(
    environment=settings.environment.value,
    log_level=settings.log_level,
    log_format=settings.log_format,
)

logger = get_logger(__name__)

# 优雅关闭配置
SHUTDOWN_TIMEOUT = 30  # 关闭超时时间（秒）
_shutdown_event = asyncio.Event()


def _signal_handler(signum: int, frame) -> None:
    """信号处理器

    处理 SIGTERM (15) 和 SIGINT (2) 信号，触发优雅关闭。

    Args:
        signum: 信号编号
        frame: 当前栈帧
    """
    logger.info(
        "signal_received",
        signal_name=signal.Signals(signum).name,
        signal_number=signum,
    )
    _shutdown_event.set()


# 注册信号处理器
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理

    处理应用启动和关闭时的资源管理，实现优雅关闭。
    """
    logger.info(
        "app_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # 启动时初始化
    # 注意：数据库连接池由 SQLAlchemy 异步引擎自动管理
    # LLM 服务和 Agent 采用延迟加载，在首次使用时初始化

    logger.info("app_started", startup_checks="passed")

    yield

    # ========== 关闭时清理 ==========
    logger.info("app_shutting_down", timeout=SHUTDOWN_TIMEOUT)

    # 创建清理任务列表
    cleanup_tasks = []

    # 1. 关闭 Redis 连接池
    async def close_redis_pool():
        try:
            from app.infra.redis import close_redis

            await close_redis()
            logger.info("redis_closed")
        except Exception as e:
            logger.error("redis_close_failed", error=str(e))

    cleanup_tasks.append(close_redis_pool())

    # 2. 关闭 LangGraph Checkpointer
    async def close_checkpointer():
        try:
            from app.agent.graph.checkpoint import close_postgres_checkpointer

            await close_postgres_checkpointer()
            logger.info("checkpointer_closed")
        except Exception as e:
            logger.error("checkpointer_close_failed", error=str(e))

    cleanup_tasks.append(close_checkpointer())

    # 3. 关闭数据库连接池
    async def close_database_pool():
        try:
            from app.infra.database import dispose_engine

            await dispose_engine()
            logger.info("database_closed")
        except Exception as e:
            logger.error("database_close_failed", error=str(e))

    cleanup_tasks.append(close_database_pool())

    # 3. 等待所有清理任务完成（带超时）
    try:
        await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=SHUTDOWN_TIMEOUT)
        logger.info("cleanup_completed")
    except asyncio.TimeoutError:
        logger.warning("cleanup_timeout", timeout=SHUTDOWN_TIMEOUT)
    except Exception as e:
        logger.error("cleanup_failed", error=str(e))

    logger.info("app_shutdown_complete")


def create_app() -> FastAPI:
    """创建 FastAPI 应用

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # SlowAPI 限流
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # ========== 添加中间件 ==========

    # 安全响应头 - 必须在最前面，确保所有响应都有安全头
    app.add_middleware(SecurityHeadersMiddleware)

    # 请求大小限制 - 在其他中间件之前检查
    app.add_middleware(MaxRequestSizeMiddleware, max_size=settings.max_request_size)

    # 速率限制中间件
    app.add_middleware(SlowAPIMiddleware)

    # 可观测性中间件
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # 租户认证中间件
    app.add_middleware(
        TenantMiddleware,
        enable_cross_tenant=settings.enable_cross_tenant,
    )

    # CORS 配置 - 允许前端跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # TrustedHost - 防止 Host Header 攻击
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )

    # GZip 压缩 - 仅生产环境
    if not settings.is_development:
        app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 注册路由
    _register_routes(app)

    # 注册异常处理器
    _register_exception_handlers(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """注册路由"""

    @app.get("/health")
    @limiter.limit(RateLimit.HEALTH)
    async def health_check(request: StarletteRequest):
        """健康检查（深度检查）

        检查应用状态和所有依赖服务的健康状态：
        - Redis（缓存）
        - PostgreSQL（数据库）
        - LLM 服务（可选）
        """
        checks = {
            "app": "healthy",
            "version": settings.app_version,
            "timestamp": None,
        }

        # 获取当前时间
        from datetime import UTC, datetime

        checks["timestamp"] = datetime.now(UTC).isoformat()

        # 1. 检查 Redis 连接
        try:
            from app.infra.redis import ping as redis_ping

            redis_status = await redis_ping()
            checks["redis"] = "healthy" if redis_status else "unhealthy"
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            checks["redis"] = "error"

        # 2. 检查 PostgreSQL 数据库连接
        try:
            from app.infra.database import check_database_health

            db_status = await check_database_health()
            checks["database"] = "healthy" if db_status else "unhealthy"
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            checks["database"] = "error"

        # 3. 计算整体健康状态
        # 所有服务都健康 → healthy
        # 有服务不健康但没有错误 → degraded
        # 有服务错误 → unhealthy
        service_statuses = [
            checks.get("redis"),
            checks.get("database"),
        ]

        if "error" in service_statuses:
            checks["status"] = "unhealthy"
        elif "unhealthy" in service_statuses:
            checks["status"] = "degraded"
        else:
            checks["status"] = "healthy"

        return checks

    @app.get("/health/ready")
    @limiter.limit(RateLimit.HEALTH)
    async def readiness_check(request: StarletteRequest):
        """就绪检查（浅度检查）

        仅检查应用本身是否就绪，不检查依赖服务。
        用于 Kubernetes readiness probe。
        """
        return {
            "ready": True,
            "version": settings.app_version,
        }

    @app.get("/health/live")
    @limiter.limit(RateLimit.HEALTH)
    async def liveness_check(request: StarletteRequest):
        """存活检查

        仅检查应用是否存活，最简单的端点。
        用于 Kubernetes liveness probe。
        """
        return {"alive": True}

    @app.get("/")
    @limiter.limit(RateLimit.API)
    async def root(request: StarletteRequest):
        """根路径"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
        }

    @app.get("/metrics")
    @limiter.limit(RateLimit.API)
    async def metrics(request: StarletteRequest):
        """Prometheus 指标"""
        from fastapi.responses import Response

        from app.observability.metrics import get_metrics_text

        return Response(
            content=get_metrics_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # 注册 API 路由
    from app.api.v1 import router as api_v1_router

    app.include_router(api_v1_router, prefix=settings.api_prefix)


def _register_exception_handlers(app: FastAPI) -> None:
    """注册异常处理器"""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: StarletteRequest, exc: Exception):
        """全局异常处理

        根据环境返回适当的错误信息：
        - 生产环境：不暴露敏感错误信息
        - 开发环境：显示详细错误
        """
        # 分类错误
        error_context = classify_error(exc)

        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
            error_category=error_context.category.value,
            error_severity=error_context.severity.value,
        )

        # 获取用户友好的错误消息
        user_message = get_user_friendly_message(exc, error_context)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": user_message,
                "category": error_context.category.value if settings.debug else None,
            },
        )


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
