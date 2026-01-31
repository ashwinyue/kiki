"""自定义 Agent 模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class CustomAgentBase(SQLModel):
    """自定义 Agent 基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    avatar: str | None = Field(default=None, max_length=64)
    is_builtin: bool = Field(default=False)


class CustomAgent(CustomAgentBase, table=True):
    """自定义 Agent 表模型

    对应 WeKnora99 的 custom_agents 表
    """

    __tablename__ = "custom_agents"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    created_by: str | None = Field(default=None, max_length=36)
    config: Any = Field(default={}, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class CustomAgentCreate(CustomAgentBase):
    """自定义 Agent 创建模型"""

    tenant_id: int
    config: Any = {}


class CustomAgentUpdate(SQLModel):
    """自定义 Agent 更新模型"""

    name: str | None = None
    description: str | None = None
    avatar: str | None = None
    config: Any | None = None


class CustomAgentPublic(CustomAgentBase):
    """自定义 Agent 公开信息"""

    id: str
    tenant_id: int
    created_at: datetime


# 向后兼容别名
Agent = CustomAgent
AgentCreate = CustomAgentCreate
AgentUpdate = CustomAgentUpdate
AgentPublic = CustomAgentPublic
