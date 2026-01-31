"""API Key 模型

对齐 WeKnora99 表结构，提供 API Key 认证功能。
"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Column, Field, SQLModel

# ============== 枚举类型 ==============


class ApiKeyType(str, Enum):
    """API Key 类型"""

    PERSONAL = "personal"  # 个人 API Key
    SERVICE = "service"  # 服务间调用 Key
    MCP = "mcp"  # MCP 服务器专用 Key
    WEBHOOK = "webhook"  # Webhook 验证 Key


class ApiKeyStatus(str, Enum):
    """API Key 状态"""

    ACTIVE = "active"  # 激活
    REVOKED = "revoked"  # 已吊销
    EXPIRED = "expired"  # 已过期


# ============== 数据库模型 ==============


class ApiKeyBase(SQLModel):
    """API Key 基础模型"""

    name: str = Field(max_length=255)
    key_prefix: str = Field(max_length=20)


class ApiKey(ApiKeyBase, table=True):
    """API Key 表模型"""

    __tablename__ = "api_keys"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    key_type: ApiKeyType = Field(default=ApiKeyType.PERSONAL)
    status: ApiKeyStatus = Field(default=ApiKeyStatus.ACTIVE)
    hashed_key: str = Field(max_length=255)
    scopes: list[str] | None = Field(
        default=None, sa_column=Column(ARRAY(String), default=None)
    )
    description: str | None = Field(default=None, max_length=500)
    rate_limit: int | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    last_used_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============== 创建/更新模型 ==============


class ApiKeyCreate(SQLModel):
    """API Key 创建模型"""

    name: str = Field(max_length=255)
    key_type: ApiKeyType = Field(default=ApiKeyType.PERSONAL)
    scopes: list[str] | None = Field(default=None)
    description: str | None = Field(default=None)
    rate_limit: int | None = Field(default=None)
    expires_in_days: int | None = Field(default=None)


class ApiKeyUpdate(SQLModel):
    """API Key 更新模型"""

    name: str | None = None
    status: ApiKeyStatus | None = None
    scopes: list[str] | None = None
    description: str | None = None
    rate_limit: int | None = None
    expires_at: datetime | None = None


# ============== 响应模型 ==============


class ApiKeyRead(SQLModel):
    """API Key 读取模型（不含敏感信息）"""

    id: int
    name: str
    key_prefix: str
    key_type: ApiKeyType
    status: ApiKeyStatus
    scopes: list[str] | None
    expires_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ApiKeyResponse(SQLModel):
    """API Key 创建响应（包含完整 Key）"""

    id: int
    name: str
    key: str  # 完整的 Key（仅创建时返回）
    key_prefix: str
    key_type: ApiKeyType
    status: ApiKeyStatus
    scopes: list[str]
    expires_at: datetime | None
    created_at: datetime


class ApiKeyVerifyResponse(SQLModel):
    """API Key 验证响应"""

    valid: bool
    key_id: int | None
    user_id: int | None
    key_type: ApiKeyType | None
    scopes: list[str]
