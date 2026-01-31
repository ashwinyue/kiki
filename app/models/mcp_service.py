"""MCP 服务模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class MCPServiceBase(SQLModel):
    """MCP 服务基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    enabled: bool = Field(default=True)
    transport_type: str = Field(max_length=50)  # stdio, sse, http


class MCPService(MCPServiceBase, table=True):
    """MCP 服务表模型"""

    __tablename__ = "mcp_services"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    url: str | None = Field(default=None, max_length=512)
    headers: Any | None = Field(default=None, sa_column=Column(JSONB))
    auth_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    advanced_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    stdio_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    env_vars: Any | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class MCPServiceCreate(MCPServiceBase):
    """MCP 服务创建模型"""

    tenant_id: int
    url: str | None = None
    headers: Any | None = None
    auth_config: Any | None = None
    stdio_config: Any | None = None
    env_vars: Any | None = None


class MCPServiceUpdate(SQLModel):
    """MCP 服务更新模型"""

    name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    url: str | None = None
    transport_type: str | None = None
    headers: Any | None = None
    auth_config: Any | None = None
    stdio_config: Any | None = None
    env_vars: Any | None = None


class MCPServicePublic(MCPServiceBase):
    """MCP 服务公开信息"""

    id: str
    tenant_id: int
    created_at: datetime
