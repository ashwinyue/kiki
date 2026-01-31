"""消息模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class MessageBase(SQLModel):
    """消息基础模型"""

    role: str = Field(max_length=50)  # user, assistant, system
    content: str


class Message(MessageBase, table=True):
    """消息表模型

    对应 WeKnora99 的 messages 表
    """

    __tablename__ = "messages"

    id: str = Field(default=None, primary_key=True, max_length=36)
    request_id: str = Field(max_length=36)
    session_id: str = Field(max_length=36, foreign_key="sessions.id")

    # 知识库引用
    knowledge_references: Any | None = Field(default=None, sa_column=Column(JSONB))
    agent_steps: Any | None = Field(default=None, sa_column=Column(JSONB))
    mentioned_items: Any | None = Field(default=None, sa_column=Column(JSONB))
    tool_calls: Any | None = Field(default=None, sa_column=Column(JSONB))

    is_completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class MessageCreate(MessageBase):
    """消息创建模型"""

    request_id: str
    session_id: str
    knowledge_references: Any | None = None
    agent_steps: Any | None = None
    mentioned_items: Any | None = None


class MessageUpdate(SQLModel):
    """消息更新模型"""

    content: str | None = None
    agent_steps: Any | None = None
    is_completed: bool | None = None


class MessagePublic(MessageBase):
    """消息公开信息"""

    id: str
    request_id: str
    session_id: str
    is_completed: bool
    created_at: datetime
