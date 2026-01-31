"""系统信息 API 路由

对齐 WeKnora99 系统信息 API 规范
"""

from fastapi import APIRouter

from app.observability.logging import get_logger
from app.schemas.response import ApiResponse
from app.schemas.system import (
    EngineInfoResponse,
    HealthResponse,
    MinioBucket,
    MinioBucketsResponse,
    StorageBucket,
    StorageInfoResponse,
    SystemInfoResponse,
)
from app.services.system_service import SystemService

router = APIRouter(prefix="/system", tags=["system"])
logger = get_logger(__name__)

# 全局系统服务实例
_system_service: SystemService | None = None


def get_system_service() -> SystemService:
    """获取系统服务实例（单例）

    Returns:
        SystemService: 系统服务实例
    """
    global _system_service
    if _system_service is None:
        _system_service = SystemService()
    return _system_service


@router.get("/info", response_model=ApiResponse[SystemInfoResponse])
async def get_system_info():
    """获取系统信息

    对齐 WeKnora99 GET /system/info

    返回版本、引擎、存储等完整系统信息。
    """
    service = get_system_service()
    info = await service.get_system_info()

    logger.info("system_info_queried")

    return ApiResponse(success=True, data=SystemInfoResponse(**info))


@router.get("/health", response_model=ApiResponse[HealthResponse])
async def health_check():
    """健康检查

    检查数据库、Redis、向量存储等组件的健康状态。

    Returns:
        健康状态信息
    """
    service = get_system_service()
    health = await service.health_check()

    logger.info("health_check_completed", status=health["status"])

    return ApiResponse(success=True, data=HealthResponse(**health))


@router.get("/engines", response_model=ApiResponse[EngineInfoResponse])
async def get_engine_info():
    """获取引擎状态

    对齐 WeKnora99 系统信息 API 中的引擎状态部分。

    返回向量存储、搜索引擎、图数据库等引擎的状态。
    """
    service = get_system_service()
    engines = await service.get_engine_info()

    logger.info("engine_info_queried", engines=list(engines.keys()))

    return ApiResponse(
        success=True,
        data=EngineInfoResponse(
            vector=engines.get("vector"),
            search=engines.get("search"),
            graph=engines.get("graph"),
        ),
    )


@router.get("/storage", response_model=ApiResponse[StorageInfoResponse])
async def get_storage_info():
    """获取存储状态

    返回 MinIO/S3/COS 等存储的存储桶信息。
    """
    service = get_system_service()
    storage = await service.get_storage_info()
    storage_type = service.settings.storage_type if hasattr(service, "settings") else "local"

    from app.config.settings import get_settings

    settings = get_settings()
    storage_type = settings.storage_type

    logger.info("storage_info_queried", storage_type=storage_type, bucket_count=len(storage))

    return ApiResponse(
        success=True,
        data=StorageInfoResponse(
            buckets={k: StorageBucket(**v.to_dict()) for k, v in storage.items()},
            storage_type=storage_type,
        ),
    )


@router.get("/minio/buckets", response_model=ApiResponse[MinioBucketsResponse])
async def list_minio_buckets():
    """列出 MinIO 存储桶

    对齐 WeKnora99 GET /system/minio/buckets

    返回所有 MinIO 存储桶及其访问权限。
    """
    service = get_system_service()
    storage = await service.get_storage_info()

    buckets = [
        MinioBucket(
            name=bucket.name,
            creation_date=bucket.created_at,
        )
        for bucket in storage.values()
    ]

    logger.info("minio_buckets_queried", count=len(buckets))

    return ApiResponse(success=True, data=MinioBucketsResponse(buckets=buckets))


__all__ = ["router"]
