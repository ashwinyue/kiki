"""会话模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class SessionBase(SQLModel):
    """会话基础模型"""

    title: str | None = Field(default=None, max_length=255)
    description: str | None = None


class Session(SessionBase, table=True):
    """会话表模型

    对应 WeKnora99 的 sessions 表
    """

    __tablename__ = "sessions"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    user_id: str | None = Field(default=None, max_length=36)

    # 知识库关联
    knowledge_base_id: str | None = Field(default=None, max_length=36)
    agent_id: str | None = Field(default=None, max_length=36)

    # 检索配置
    max_rounds: int = Field(default=5)
    enable_rewrite: bool = Field(default=True)
    fallback_strategy: str = Field(default="fixed", max_length=255)
    fallback_response: str = Field(default="很抱歉，我暂时无法回答这个问题。")
    keyword_threshold: float = Field(default=0.5)
    vector_threshold: float = Field(default=0.5)

    # 重排序配置
    rerank_model_id: str | None = Field(default=None, max_length=64)
    embedding_top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=10)
    rerank_threshold: float = Field(default=0.65)

    # 摘要配置
    summary_model_id: str | None = Field(default=None, max_length=64)
    summary_parameters: Any | None = Field(default=None, sa_column=Column(JSONB))

    # Agent 和上下文配置
    agent_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    context_config: Any | None = Field(default=None, sa_column=Column(JSONB))

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class SessionCreate(SessionBase):
    """会话创建模型"""

    tenant_id: int
    user_id: str | None = None
    knowledge_base_id: str | None = None
    agent_id: str | None = None
    agent_config: Any | None = None
    context_config: Any | None = None


class SessionUpdate(SQLModel):
    """会话更新模型"""

    title: str | None = None
    description: str | None = None
    knowledge_base_id: str | None = None
    agent_id: str | None = None
    agent_config: Any | None = None


class SessionPublic(SessionBase):
    """会话公开信息"""

    id: str
    tenant_id: int
    user_id: str | None
    knowledge_base_id: str | None
    agent_id: str | None
    created_at: datetime


# 向后兼容
ChatSession = Session
