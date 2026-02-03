"""系统信息服务

提供系统版本、引擎状态、存储状态等信息的查询功能。

对齐 WeKnora99 系统信息 API 规范。
"""

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.observability.logging import get_logger

# ElasticsearchService 已移除（RAG 相关）

logger = get_logger(__name__)
settings = get_settings()


class SystemInfo:
    """系统信息"""

    version: str
    commit_id: str
    build_time: str
    python_version: str

    def __init__(
        self,
        version: str,
        commit_id: str = "",
        build_time: str = "",
        python_version: str = "",
    ):
        self.version = version
        self.commit_id = commit_id or "unknown"
        self.build_time = build_time or "unknown"
        self.python_version = python_version

    def to_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "commit_id": self.commit_id,
            "build_time": self.build_time,
            "python_version": self.python_version,
        }


class EngineStatus:
    """引擎状态"""

    name: str
    enabled: bool
    healthy: bool
    details: dict[str, str | bool | int]

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        healthy: bool = False,
        details: dict[str, str | bool | int] | None = None,
    ):
        self.name = name
        self.enabled = enabled
        self.healthy = healthy
        self.details = details or {}

    def to_dict(self) -> dict[str, str | bool | int | dict[str, str | bool | int]]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "details": self.details,
        }


class StorageBucket:
    """存储桶信息"""

    name: str
    policy: str  # "public", "private", "custom"
    created_at: str | None

    def __init__(self, name: str, policy: str = "private", created_at: str | None = None):
        self.name = name
        self.policy = policy
        self.created_at = created_at

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "policy": self.policy,
            "created_at": self.created_at,
        }


class SystemService:
    """系统信息服务

    提供系统信息查询、引擎状态检查、存储状态查询等功能。
    """

    def __init__(self):
        """初始化系统信息服务"""
        self._version = os.getenv("SYSTEM_VERSION", settings.app_version)
        self._commit_id = os.getenv("COMMIT_ID", "unknown")
        self._build_time = os.getenv("BUILD_TIME", "unknown")

    async def get_version_info(self) -> SystemInfo:
        """获取版本信息

        Returns:
            SystemInfo: 版本信息
        """
        return SystemInfo(
            version=self._version,
            commit_id=self._commit_id,
            build_time=self._build_time,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

    async def get_engine_info(self) -> dict[str, EngineStatus]:
        """获取引擎状态

        Returns:
            引擎状态字典
        """
        result: dict[str, EngineStatus] = {}

        # 向量存储引擎
        vector_engine = await self._get_vector_engine_status()
        result["vector"] = vector_engine

        # 搜索引擎
        search_engine = await self._get_search_engine_status()
        result["search"] = search_engine

        # 图数据库引擎 (Neo4j - 暂未实现)
        result["graph"] = EngineStatus(
            name="Neo4j",
            enabled=False,
            healthy=False,
            details={"status": "未启用"},
        )

        logger.info("engine_info_queried", engines=list(result.keys()))
        return result

    async def _get_vector_engine_status(self) -> EngineStatus:
        """获取向量存储引擎状态

        Returns:
            EngineStatus: 向量引擎状态
        """
        vector_type = settings.vector_store_type
        enabled = vector_type != "memory"
        healthy = False
        details: dict[str, str | bool] = {
            "type": vector_type,
            "dimension": settings.embedding_dimensions,
        }

        if enabled:
            try:
                from app.services.shared.vector_service import VectorService

                service = VectorService()
                is_healthy = await service.health_check()
                healthy = is_healthy
                details["healthy"] = is_healthy
            except Exception as e:
                details["error"] = str(e)
                logger.warning("vector_engine_health_check_failed", error=str(e))

        return EngineStatus(
            name=self._get_vector_engine_name(vector_type),
            enabled=enabled,
            healthy=healthy,
            details=details,
        )

    def _get_vector_engine_name(self, vector_type: str) -> str:
        """获取向量引擎名称

        Args:
            vector_type: 向量存储类型

        Returns:
            引擎名称
        """
        names = {
            "qdrant": "Qdrant",
            "pgvector": "PGVector",
            "pinecone": "Pinecone",
            "chroma": "Chroma",
            "memory": "内存存储",
            "elasticsearch": "Elasticsearch",
        }
        return names.get(vector_type, vector_type)

    async def _get_search_engine_status(self) -> EngineStatus:
        """获取搜索引擎状态

        Returns:
            EngineStatus: 搜索引擎状态
        """
        # Elasticsearch 已移除（RAG 相关）
        return EngineStatus(
            name="未配置",
            enabled=False,
            healthy=False,
            details={"type": "Elasticsearch 已移除"},
        )

    async def get_storage_info(self) -> dict[str, StorageBucket]:
        """获取存储状态

        Returns:
            存储桶信息字典
        """
        storage_type = settings.storage_type
        result: dict[str, StorageBucket] = {}

        if storage_type == "minio":
            buckets = await self._list_minio_buckets()
            for bucket in buckets:
                result[bucket.name] = bucket
        elif storage_type == "local":
            # 本地存储
            result["local"] = StorageBucket(
                name="local",
                policy="private",
                created_at=None,
            )
        elif storage_type == "cos":
            result["cos"] = StorageBucket(
                name=settings.cos_bucket_name,
                policy="private",
                created_at=None,
            )

        logger.info(
            "storage_info_queried",
            storage_type=storage_type,
            bucket_count=len(result),
        )
        return result

    async def _list_minio_buckets(self) -> list[StorageBucket]:
        """列出 MinIO 存储桶

        Returns:
            存储桶列表
        """
        if not self._is_minio_enabled():
            logger.warning("minio_not_enabled")
            return []

        try:
            from minio import Minio

            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key_id,
                secret_key=settings.minio_secret_access_key,
                secure=settings.minio_use_ssl,
            )

            buckets = client.list_buckets()
            result = []

            for bucket in buckets:
                policy = "private"
                try:
                    policy_str = client.get_bucket_policy(bucket.name)
                    if policy_str:
                        policy = self._parse_bucket_policy(policy_str)
                except Exception:
                    # 无策略表示私有
                    logger.debug("minio_bucket_no_policy", bucket=bucket.name)

                result.append(
                    StorageBucket(
                        name=bucket.name,
                        policy=policy,
                        created_at=bucket.creation_date.isoformat()
                        if bucket.creation_date
                        else None,
                    )
                )

            logger.info("minio_buckets_listed", count=len(result))
            return result

        except ImportError:
            logger.warning("minio_package_not_installed")
            return []
        except Exception as e:
            logger.error("minio_list_buckets_failed", error=str(e))
            return []

    def _is_minio_enabled(self) -> bool:
        """检查 MinIO 是否启用

        Returns:
            是否启用
        """
        return bool(
            settings.storage_type == "minio"
            and settings.minio_endpoint
            and settings.minio_access_key_id
            and settings.minio_secret_access_key,
        )

    def _parse_bucket_policy(self, policy_str: str) -> str:
        """解析存储桶策略

        Args:
            policy_str: 策略 JSON 字符串

        Returns:
            策略类型: public, private, custom
        """
        import json

        try:
            policy = json.loads(policy_str)
            statements = policy.get("Statement", [])

            for stmt in statements:
                if stmt.get("Effect") != "Allow":
                    continue

                principal = stmt.get("Principal")
                if principal == "*" or (
                    isinstance(principal, dict)
                    and principal.get("AWS") in ("*", ["*"])
                ):
                    action = stmt.get("Action")
                    actions = action if isinstance(action, list) else [action]

                    for a in actions:
                        a_lower = a.lower() if isinstance(a, str) else ""
                        if a_lower in ("s3:getobject", "s3:*", "*"):
                            return "public"

            return "custom"

        except (json.JSONDecodeError, KeyError, TypeError):
            return "custom"

    async def health_check(self) -> dict[str, Any]:
        """健康检查

        Returns:
            健康状态信息
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "checks": {},
        }

        # 数据库健康检查
        try:
            from app.infra.database import get_db_session

            async with get_db_session() as session:
                await session.execute("SELECT 1")
            health_status["checks"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "unhealthy"

        # Redis 健康检查
        try:
            from app.infra.redis import get_redis_client

            redis_client = await get_redis_client()
            await redis_client.ping()
            health_status["checks"]["redis"] = {"status": "healthy"}
        except Exception as e:
            health_status["checks"]["redis"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        # 向量存储健康检查
        try:
            from app.services.shared.vector_service import VectorService

            vector_service = VectorService()
            vector_healthy = await vector_service.health_check()
            health_status["checks"]["vector_store"] = {
                "status": "healthy" if vector_healthy else "unhealthy",
                "type": settings.vector_store_type,
            }
            if not vector_healthy and health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"

        logger.info("health_check_completed", status=health_status["status"])
        return health_status

    async def get_system_info(self) -> dict[str, Any]:
        """获取完整系统信息

        Returns:
            完整系统信息字典
        """
        version_info = await self.get_version_info()
        engine_info = await self.get_engine_info()
        storage_info = await self.get_storage_info()

        return {
            "version": version_info.to_dict(),
            "engines": {k: v.to_dict() for k, v in engine_info.items()},
            "storage": {k: v.to_dict() for k, v in storage_info.items()},
            "environment": {
                "name": settings.environment.value,
                "platform": platform.system(),
                "python_version": version_info.python_version,
                "debug": settings.debug,
            },
            "minio_enabled": self._is_minio_enabled(),
        }


__all__ = ["SystemService"]
