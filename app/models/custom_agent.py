"""自定义 Agent 模型

Multi-Agent 架构支持，参考 WeKnora99 和 LangGraph 最佳实践。
"""

from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.timestamp import TimestampMixin


# ============== Agent 角色常量 ==============


class AgentRole:
    """Agent 角色常量"""

    SUPERVISOR = "supervisor"  # 协调者，管理多个 worker
    WORKER = "worker"  # 工作者，执行具体任务
    ROUTER = "router"  # 路由器，决定调用哪个 agent
    LEAF = "leaf"  # 叶子节点，不调用其他 agent


class GraphType:
    """图类型常量"""

    SINGLE = "single"  # 单 Agent
    SUPERVISOR = "supervisor"  # Supervisor 模式
    ROUTER = "router"  # Router 模式
    HIERARCHICAL = "hierarchical"  # 分层模式


class CustomAgentBase(SQLModel):
    """自定义 Agent 基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    avatar: str | None = Field(default=None, max_length=64)
    is_builtin: bool = Field(default=False)


class CustomAgent(TimestampMixin, CustomAgentBase, table=True):
    """自定义 Agent 表模型（Multi-Agent 支持）

    对应 WeKnora99 的 custom_agents 表，扩展支持 Multi-Agent 架构。
    """

    __tablename__ = "custom_agents"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    created_by: str | None = Field(default=None, max_length=36)
    config: Any = Field(default={}, sa_column=Column(JSONB))
    deleted_at: datetime | None = Field(default=None)

    # ========== Multi-Agent 配置 ==========
    # Agent 角色：supervisor, worker, router, leaf
    agent_role: str = Field(
        default="leaf",
        max_length=50,
        description="Agent 角色: supervisor, worker, router, leaf",
    )

    # 父 Agent ID（hierarchical 模式使用）
    parent_agent_id: str | None = Field(
        default=None,
        max_length=64,
        description="父 Agent ID（hierarchical 模式使用）",
    )

    # 允许调用的 Worker 列表（supervisor 模式使用）
    allowed_workers: list[str] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="允许调用的 worker agent ID 列表（JSONB）",
    )


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
