"""系统信息 Schema

对齐 WeKnora99 系统信息 API 规范
"""

from typing import Any

from pydantic import BaseModel, Field


class VersionInfo(BaseModel):
    """版本信息"""

    version: str = Field(..., description="系统版本")
    commit_id: str = Field(..., alias="commit_id", description="Git 提交 ID")
    build_time: str = Field(..., alias="build_time", description="构建时间")
    python_version: str = Field(..., alias="python_version", description="Python 版本")

    class Config:
        populate_by_name = True


class EngineDetails(BaseModel):
    """引擎详细信息"""

    type: str | None = Field(None, description="引擎类型")
    dimension: int | None = Field(None, description="向量维度")
    healthy: bool | None = Field(None, description="是否健康")
    status: str | None = Field(None, description="状态描述")
    cluster_name: str | None = Field(None, alias="cluster_name", description="集群名称")
    version: str | None = Field(None, description="引擎版本")
    error: str | None = Field(None, description="错误信息")

    class Config:
        populate_by_name = True


class EngineStatus(BaseModel):
    """引擎状态"""

    name: str = Field(..., description="引擎名称")
    enabled: bool = Field(..., description="是否启用")
    healthy: bool = Field(..., description="是否健康")
    details: EngineDetails = Field(default_factory=EngineDetails, description="详细信息")


class StorageBucket(BaseModel):
    """存储桶信息"""

    name: str = Field(..., description="存储桶名称")
    policy: str = Field(..., description="访问策略: public, private, custom")
    created_at: str | None = Field(None, alias="created_at", description="创建时间")

    class Config:
        populate_by_name = True


class EnvironmentInfo(BaseModel):
    """环境信息"""

    name: str = Field(..., description="环境名称: development, staging, production")
    platform: str = Field(..., description="操作系统")
    python_version: str = Field(..., alias="python_version", description="Python 版本")
    debug: bool = Field(..., description="调试模式")

    class Config:
        populate_by_name = True


class SystemInfoResponse(BaseModel):
    """完整系统信息响应"""

    version: VersionInfo = Field(..., description="版本信息")
    engines: dict[str, EngineStatus] = Field(default_factory=dict, description="引擎状态")
    storage: dict[str, StorageBucket] = Field(
        default_factory=dict, description="存储状态"
    )
    environment: EnvironmentInfo = Field(..., description="环境信息")
    minio_enabled: bool = Field(..., alias="minio_enabled", description="MinIO 是否启用")

    class Config:
        populate_by_name = True


class HealthCheck(BaseModel):
    """健康检查项"""

    status: str = Field(..., description="状态: healthy, unhealthy")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="整体状态: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="检查时间")
    checks: dict[str, HealthCheck | dict[str, Any]] = Field(
        default_factory=dict, description="各组件检查结果"
    )


class EngineInfoResponse(BaseModel):
    """引擎信息响应"""

    vector: EngineStatus | None = Field(None, description="向量存储引擎")
    search: EngineStatus | None = Field(None, description="搜索引擎")
    graph: EngineStatus | None = Field(None, description="图数据库引擎")


class StorageInfoResponse(BaseModel):
    """存储信息响应"""

    buckets: dict[str, StorageBucket] = Field(
        default_factory=dict, description="存储桶列表"
    )
    storage_type: str = Field(..., alias="storage_type", description="存储类型")


class MinioBucket(BaseModel):
    """MinIO 存储桶信息（兼容旧版）"""

    name: str = Field(..., description="存储桶名称")
    creation_date: str | None = Field(None, alias="creationDate", description="创建日期")

    class Config:
        populate_by_name = True


class MinioBucketsResponse(BaseModel):
    """MinIO 存储桶列表响应（兼容旧版）"""

    buckets: list[MinioBucket] = Field(default_factory=list)


__all__ = [
    "VersionInfo",
    "EngineStatus",
    "EngineDetails",
    "StorageBucket",
    "EnvironmentInfo",
    "SystemInfoResponse",
    "HealthResponse",
    "EngineInfoResponse",
    "StorageInfoResponse",
    "MinioBucket",
    "MinioBucketsResponse",
]
