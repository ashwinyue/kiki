"""应用入口

创建 FastAPI 应用并配置所有组件。
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import ObservabilityMiddleware, RequestContextMiddleware


# 获取配置
settings = get_settings()

# 配置日志
configure_logging(
    environment=settings.environment.value,
    log_level=settings.log_level,
    log_format=settings.log_format,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理"""
    logger.info(
        "app_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # 启动时初始化
    # 1. 初始化 LLM 服务（延迟加载，在首次使用时初始化）
    # from app.core.llm import get_llm_service
    # llm_service = get_llm_service()

    # 2. 初始化检查点保存器（延迟加载，在 Agent 中初始化）

    # 3. 初始化 Langfuse 客户端（如果配置了）
    # if settings.langfuse_public_key and settings.langfuse_secret_key:
    #     from langfuse import Langfuse
    #     langfuse = Langfuse(
    #         public_key=settings.langfuse_public_key,
    #         secret_key=settings.langfuse_secret_key,
    #         host=settings.langfuse_host,
    #     )
    #     logger.info("langfuse_initialized")

    yield

    # 关闭时清理
    logger.info("app_shutting_down")

    # 关闭 Redis 连接池
    from app.core.redis import close_redis
    await close_redis()

    # 关闭 Agent
    # from app.core.agent import get_agent
    # agent = await get_agent()
    # await agent.close()


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

    # 添加中间件
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(RequestContextMiddleware)

    # 注册路由
    _register_routes(app)

    # 注册异常处理器
    _register_exception_handlers(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """注册路由"""

    @app.get("/health")
    async def health_check():
        """健康检查

        检查应用状态和依赖服务（Redis）的健康状态。
        """
        from app.core.redis import ping as redis_ping

        checks = {
            "app": "healthy",
            "version": settings.app_version,
        }

        # 检查 Redis 连接
        redis_status = await redis_ping()
        checks["redis"] = "healthy" if redis_status else "unhealthy"

        # 如果有依赖服务不健康，整体状态为 degraded
        if not redis_status:
            checks["status"] = "degraded"
        else:
            checks["status"] = "healthy"

        return checks

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
        }

    # 注册 API 路由
    from app.api.v1 import router as api_v1_router
    app.include_router(api_v1_router, prefix=settings.api_prefix)


def _register_exception_handlers(app: FastAPI) -> None:
    """注册异常处理器"""

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """全局异常处理"""
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc) if settings.debug else "An error occurred",
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
