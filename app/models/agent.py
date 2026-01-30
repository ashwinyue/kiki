"""Agent 数据模型

提供 Agent、Tool、PromptTemplate、AgentExecution 等持久化模型。
"""

from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional, List, TYPE_CHECKING

from sqlmodel import Field, Relationship, SQLModel, Column
from sqlalchemy import JSON


# ============== 枚举定义 ==============

class AgentType(str, Enum):
    """Agent 类型"""

    SINGLE = "single"  # 单一 Agent
    ROUTER = "router"  # 路由 Agent
    SUPERVISOR = "supervisor"  # 监督 Agent
    WORKER = "worker"  # 工作 Agent
    HANDOFF = "handoff"  # 切换 Agent


class AgentStatus(str, Enum):
    """Agent 状态"""

    ACTIVE = "active"  # 激活
    DISABLED = "disabled"  # 禁用
    DELETED = "deleted"  # 已删除


# ============== Agent 模型 ==============

class AgentBase(SQLModel):
    """Agent 基础模型"""

    name: str = Field(max_length=100, description="Agent 名称")
    description: Optional[str] = Field(default=None, max_length=500, description="Agent 描述")
    agent_type: AgentType = Field(default=AgentType.SINGLE, description="Agent 类型")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Agent 状态")
    model_name: str = Field(default="gpt-4o-mini", max_length=50, description="使用的模型")
    system_prompt: str = Field(default="", description="系统提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大生成 tokens")
    config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="额外配置")


class Agent(AgentBase, table=True):
    """Agent 表模型"""

    __tablename__ = "agents"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentCreate(AgentBase):
    """Agent 创建模型"""

    tool_ids: Optional[list[int]] = Field(default=None, description="关联的工具 ID 列表")


class AgentUpdate(SQLModel):
    """Agent 更新模型"""

    name: Optional[str] = None
    description: Optional[str] = None
    agent_type: Optional[AgentType] = None
    status: Optional[AgentStatus] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    config: Optional[dict[str, Any]] = None
    tool_ids: Optional[list[int]] = None


class AgentPublic(AgentBase):
    """Agent 公开信息"""

    id: int
    created_at: datetime


# ============== Tool 模型 ==============

class ToolBase(SQLModel):
    """工具基础模型"""

    name: str = Field(max_length=100, description="工具名称")
    description: str = Field(max_length=500, description="工具描述")
    type: str = Field(max_length=50, description="工具类型：function/python/mcp 等")
    config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="工具配置")
    is_active: bool = Field(default=True, description="是否启用")


class Tool(ToolBase, table=True):
    """工具表模型"""

    __tablename__ = "tools"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCreate(ToolBase):
    """工具创建模型"""


class ToolUpdate(SQLModel):
    """工具更新模型"""

    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None


class ToolPublic(ToolBase):
    """工具公开信息"""

    id: int
    created_at: datetime


# ============== Agent-Tool 关联表 ==============

class AgentTool(SQLModel, table=True):
    """Agent-Tool 关联表"""

    __tablename__ = "agent_tools"

    agent_id: int = Field(foreign_key="agents.id", primary_key=True)
    tool_id: int = Field(foreign_key="tools.id", primary_key=True)
    enabled: bool = Field(default=True, description="是否启用")
    config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="覆盖配置")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============== PromptTemplate 模型 ==============

class PromptTemplateBase(SQLModel):
    """Prompt 模板基础模型"""

    name: str = Field(max_length=100, description="模板名称")
    description: Optional[str] = Field(default=None, max_length=500, description="模板描述")
    category: Optional[str] = Field(default=None, max_length=50, description="分类")
    template: str = Field(description="模板内容（支持 Jinja2）")
    variables: Optional[list[str]] = Field(default=None, sa_column=Column(JSON), description="变量列表")
    is_active: bool = Field(default=True, description="是否启用")


class PromptTemplate(PromptTemplateBase, table=True):
    """Prompt 模板表模型"""

    __tablename__ = "prompt_templates"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PromptTemplateCreate(PromptTemplateBase):
    """Prompt 模板创建模型"""


class PromptTemplateUpdate(SQLModel):
    """Prompt 模板更新模型"""

    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    template: Optional[str] = None
    variables: Optional[list[str]] = None
    is_active: Optional[bool] = None


class PromptTemplatePublic(PromptTemplateBase):
    """Prompt 模板公开信息"""

    id: int
    created_at: datetime


# ============== AgentExecution 模型 ==============

class AgentExecutionBase(SQLModel):
    """Agent 执行基础模型"""

    session_id: str = Field(max_length=255, description="会话 ID")
    agent_id: int = Field(description="Agent ID")
    input_data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="输入数据")
    output_data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="输出数据")
    status: str = Field(default="running", max_length=50, description="状态：running/success/error")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    tokens_used: Optional[int] = Field(default=None, description="使用的 tokens")
    duration_ms: Optional[int] = Field(default=None, description="执行耗时（毫秒）")
    extra_data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON), description="额外元数据")


class AgentExecution(AgentExecutionBase, table=True):
    """Agent 执行历史表模型"""

    __tablename__ = "agent_executions"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentExecutionCreate(AgentExecutionBase):
    """Agent 执行创建模型"""


class AgentExecutionPublic(AgentExecutionBase):
    """Agent 执行公开信息"""

    id: int
    created_at: datetime
