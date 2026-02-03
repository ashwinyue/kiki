"""Agent 配置模型（DeerFlow 风格）

Agent 配置定义，参考 DeerFlow 和 WeKnora99 设计。

表名说明：
- agent_configs: Agent 配置表（定义）
- agent_executions: Agent 执行记录表（运行时）

设计原则：
- 配置驱动：所有 Agent 配置集中管理
- 内置 Agent 存储在代码 AGENT_REGISTRY 中
- 自定义 Agent 存储在数据库 agent_configs 表中
"""

from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.models.timestamp import TimestampMixin


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


class AgentConfigBase(SQLModel):
    """Agent 配置基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    avatar: str | None = Field(default=None, max_length=64)
    is_builtin: bool = Field(default=False)


class AgentConfig(TimestampMixin, AgentConfigBase, table=True):
    """Agent 配置表模型（DeerFlow 风格）

    存储 Agent 的配置定义，与 AGENT_REGISTRY（代码中的内置配置）对应。

    config 字段存储结构（DeerFlow 风格）：
        {
            "agent_type": "planner",      # Agent 类型
            "prompt_template": "planner", # 提示词模板
            "tools": [],                  # 工具列表
            "llm_type": "reasoning"       # LLM 类型
        }
    """

    __tablename__ = "agent_configs"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    created_by: str | None = Field(default=None, max_length=36)
    config: Any = Field(default={}, sa_column=Column(JSONB))
    deleted_at: datetime | None = Field(default=None)

    agent_role: str = Field(
        default="leaf",
        max_length=50,
        description="Agent 角色: supervisor, worker, router, leaf",
    )

    parent_agent_id: str | None = Field(
        default=None,
        max_length=64,
        description="父 Agent ID（hierarchical 模式使用）",
    )

    allowed_workers: list[str] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="允许调用的 worker agent ID 列表（JSONB）",
    )


class AgentConfigCreate(AgentConfigBase):
    """Agent 配置创建模型"""

    tenant_id: int
    config: Any = {}


class AgentConfigUpdate(SQLModel):
    """Agent 配置更新模型"""

    name: str | None = None
    description: str | None = None
    avatar: str | None = None
    config: Any | None = None


class AgentConfigPublic(AgentConfigBase):
    """Agent 配置公开信息"""

    id: str
    tenant_id: int
    created_at: datetime


__all__ = [
    "AgentConfig",
    "AgentConfigBase",
    "AgentConfigCreate",
    "AgentConfigUpdate",
    "AgentConfigPublic",
    "AgentRole",
    "GraphType",
]
