"""租户模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class TenantBase(SQLModel):
    """租户基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    api_key: str = Field(max_length=64)
    status: str = Field(default="active", max_length=50)
    business: str = Field(max_length=255)
    storage_quota: int = Field(default=10737418240)  # 10GB
    storage_used: int = Field(default=0)


class Tenant(TenantBase, table=True):
    """租户表模型"""

    __tablename__ = "tenants"

    id: int | None = Field(default=None, primary_key=True)
    retriever_engines: Any | None = Field(default=None, sa_column=Column(JSONB))
    agent_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    context_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    conversation_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    web_search_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class TenantCreate(SQLModel):
    """租户创建模型"""

    name: str
    description: str | None = None
    business: str
    api_key: str | None = None
    status: str = "active"


class TenantUpdate(SQLModel):
    """租户更新模型"""

    name: str | None = None
    description: str | None = None
    status: str | None = None
    agent_config: Any | None = None
    context_config: Any | None = None
    conversation_config: Any | None = None
    web_search_config: Any | None = None


class TenantPublic(TenantBase):
    """租户公开信息"""

    id: int
    created_at: datetime
