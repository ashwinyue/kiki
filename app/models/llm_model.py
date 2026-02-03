"""LLM 模型相关数据模型

管理 LLM 模型的配置和元数据。
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class ModelBase(SQLModel):
    """模型基础模型"""

    name: str = Field(max_length=255)
    type: str = Field(max_length=50)  # Embedding, Rerank, KnowledgeQA, VLLM, Chat
    source: str = Field(max_length=50)  # local, remote, aliyun, zhipu, openai
    description: str | None = None
    is_default: bool = Field(default=False)
    is_builtin: bool = Field(default=False)
    status: str = Field(default="active", max_length=50)


class Model(ModelBase, table=True):
    """模型表模型"""

    __tablename__ = "models"

    id: str = Field(default=None, primary_key=True, max_length=64)
    tenant_id: int
    parameters: Any = Field(default={}, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class ModelCreate(ModelBase):
    """模型创建模型"""

    tenant_id: int
    parameters: Any = {}


class ModelUpdate(SQLModel):
    """模型更新模型"""

    name: str | None = None
    type: str | None = None
    source: str | None = None
    description: str | None = None
    is_default: bool | None = None
    is_builtin: bool | None = None
    status: str | None = None
    parameters: Any | None = None


__all__ = [
    "Model",
    "ModelBase",
    "ModelCreate",
    "ModelUpdate",
]
