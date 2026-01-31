"""API Key 数据模型

支持 API Key 认证，用于服务间调用和 MCP 服务器访问。
"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime, Index, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Column, Field, SQLModel

# BaseModel should be SQLModel from sqlmodel, not from database
BaseModel = SQLModel


class ApiKeyStatus(str, Enum):
    """API Key 状态"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ApiKeyType(str, Enum):
    """API Key 类型"""

    PERSONAL = "personal"  # 个人 API Key
    SERVICE = "service"  # 服务间调用
    MCP = "mcp"  # MCP 服务器专用
    WEBHOOK = "webhook"  # Webhook 验证


class ApiKeyBase(SQLModel):
    """API Key 基础模型"""

    name: str = Field(max_length=100, description="API Key 名称")
    key_prefix: str = Field(max_length=20, description="API Key 前缀（用于显示）")
    key_type: ApiKeyType = Field(default=ApiKeyType.PERSONAL, description="API Key 类型")
    status: ApiKeyStatus = Field(default=ApiKeyStatus.ACTIVE, description="API Key 状态")
    user_id: int = Field(index=True, description="关联的用户 ID")
    scopes: list[str] = Field(default_factory=list, sa_column=Column(ARRAY(Text)), description="权限范围")
    expires_at: datetime | None = Field(default=None, description="过期时间")
    last_used_at: datetime | None = Field(default=None, description="最后使用时间")
    rate_limit: int | None = Field(default=None, description="速率限制（每分钟请求数）")


class ApiKey(ApiKeyBase, BaseModel, table=True):
    """API Key 表模型"""

    __tablename__ = "api_keys"

    id: int | None = Field(default=None, primary_key=True)

    # 存储加密后的完整 Key
    hashed_key: str = Field(max_length=255, description="加密后的 API Key")

    # 元数据
    description: str | None = Field(default=None, sa_column=Column(Text), description="描述")
    extra_data: dict | None = Field(default=None, sa_column=Column(JSONB), description="扩展元数据")

    # 索引
    __table_args__ = (
        Index("ix_api_keys_user_status", "user_id", "status"),
        Index("ix_api_keys_key_type", "key_type"),
        Index("ix_api_keys_expires_at", "expires_at"),
    )


class ApiKeyCreate(SQLModel):
    """创建 API Key 请求模型"""

    name: str = Field(..., max_length=100, description="API Key 名称")
    key_type: ApiKeyType = Field(default=ApiKeyType.PERSONAL, description="API Key 类型")
    scopes: list[str] = Field(default_factory=list, description="权限范围")
    expires_in_days: int | None = Field(default=None, description="有效期（天数）")
    description: str | None = Field(default=None, description="描述")
    rate_limit: int | None = Field(default=None, description="速率限制（每分钟请求数）")


class ApiKeyUpdate(SQLModel):
    """更新 API Key 请求模型"""

    name: str | None = Field(default=None, max_length=100)
    status: ApiKeyStatus | None = Field(default=None)
    scopes: list[str] | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)


class ApiKeyRead(SQLModel):
    """API Key 读取模型"""

    id: int
    name: str
    key_prefix: str
    key_type: ApiKeyType
    status: ApiKeyStatus
    scopes: list[str]
    expires_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ApiKeyResponse(SQLModel):
    """创建 API Key 响应模型（仅返回一次完整 Key）"""

    id: int
    name: str
    key: str  # 完整的 API Key（仅创建时返回）
    key_prefix: str
    key_type: ApiKeyType
    status: ApiKeyStatus
    scopes: list[str]
    expires_at: datetime | None
    created_at: datetime


class ApiKeyVerifyResponse(SQLModel):
    """API Key 验证响应"""

    valid: bool
    api_key_id: int | None
    user_id: int | None
    scopes: list[str]
    key_type: ApiKeyType
