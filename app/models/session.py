"""会话模型

Multi-Agent 架构支持，对齐 LangGraph 最佳实践。
"""

from typing import Any, TYPE_CHECKING
from datetime import datetime

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Relationship, SQLModel

from app.models.timestamp import TimestampMixin

if TYPE_CHECKING:
    from app.models.agent_execution import AgentExecution


class SessionBase(SQLModel):
    """会话基础模型"""

    title: str | None = Field(default=None, max_length=255)
    description: str | None = None


class Session(TimestampMixin, SessionBase, table=True):
    """会话表模型（Multi-Agent 支持）

    对应 WeKnora99 的 sessions 表，扩展支持 Multi-Agent 架构。
    """

    __tablename__ = "sessions"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    user_id: str | None = Field(default=None, max_length=36)

    # 知识库关联
    knowledge_base_id: str | None = Field(default=None, max_length=36)
    agent_id: str | None = Field(default=None, max_length=36)

    # ========== Multi-Agent 配置 ==========
    # 图类型：single, supervisor, router, hierarchical
    graph_type: str = Field(
        default="single",
        max_length=50,
        description="图类型: single, supervisor, router, hierarchical",
    )

    # 主要 Agent ID（single 模式使用）
    primary_agent_id: str | None = Field(
        default=None,
        max_length=64,
        description="主要 Agent ID（single 模式使用）",
    )

    # Supervisor 配置（supervisor 模式使用）
    supervisor_config: Any | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Supervisor 配置（supervisor 模式使用）",
    )

    # ========== 保留的兼容字段（建议移至 CustomAgent） ==========
    # TODO: 这些字段未来应移至 CustomAgent.config
    max_rounds: int = Field(default=5)
    enable_rewrite: bool = Field(default=True)
    fallback_strategy: str = Field(default="fixed", max_length=255)
    fallback_response: str = Field(default="很抱歉，我暂时无法回答这个问题。")
    keyword_threshold: float = Field(default=0.5)
    vector_threshold: float = Field(default=0.5)
    rerank_model_id: str | None = Field(default=None, max_length=64)
    embedding_top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=10)
    rerank_threshold: float = Field(default=0.65)
    summary_model_id: str | None = Field(default=None, max_length=64)
    summary_parameters: Any | None = Field(default=None, sa_column=Column(JSONB))
    agent_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    context_config: Any | None = Field(default=None, sa_column=Column(JSONB))

    deleted_at: datetime | None = Field(default=None)


class SessionCreate(SessionBase):
    """会话创建模型"""

    tenant_id: int
    user_id: str | None = None
    knowledge_base_id: str | None = None
    agent_id: str | None = None
    agent_config: Any | None = None
    context_config: Any | None = None

    # Multi-Agent 配置
    graph_type: str = "single"
    primary_agent_id: str | None = None
    supervisor_config: Any | None = None


class SessionUpdate(SQLModel):
    """会话更新模型"""

    title: str | None = None
    description: str | None = None
    knowledge_base_id: str | None = None
    agent_id: str | None = None
    agent_config: Any | None = None

    # Multi-Agent 配置
    graph_type: str | None = None
    primary_agent_id: str | None = None
    supervisor_config: Any | None = None


class SessionPublic(SessionBase):
    """会话公开信息"""

    id: str
    tenant_id: int
    user_id: str | None
    knowledge_base_id: str | None
    agent_id: str | None
    created_at: datetime

    # Multi-Agent 配置
    graph_type: str


# 向后兼容
ChatSession = Session
