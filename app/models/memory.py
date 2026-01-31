"""长期记忆模型

用于 LangGraph Store 持久化，存储跨会话的长期记忆。
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class MemoryBase(SQLModel):
    """长期记忆基础模型"""

    namespace: str = Field(max_length=255, index=True)
    key: str = Field(max_length=500)


class Memory(MemoryBase, table=True):
    """长期记忆表模型（LangGraph Store 持久化）

    用于存储跨会话的长期记忆，如用户偏好、对话摘要等。
    """

    __tablename__ = "memories"

    # 联合主键
    namespace: str = Field(max_length=255, primary_key=True)
    key: str = Field(max_length=500, primary_key=True)
    # 基础字段
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 存储值（JSONB）
    value: Any | None = Field(default=None, sa_column=Column(JSONB))
    # 过期时间（可选）
    expires_at: datetime | None = Field(default=None)


class MemoryCreate(MemoryBase):
    """记忆创建模型"""

    value: Any


class MemoryUpdate(SQLModel):
    """记忆更新模型"""

    value: Any


class MemoryPublic(MemoryBase):
    """记忆公开信息"""

    value: Any
    created_at: datetime
    updated_at: datetime


__all__ = [
    "Memory",
    "MemoryBase",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryPublic",
]
