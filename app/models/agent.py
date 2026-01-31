"""Agent 数据模型

提供 Agent、PromptTemplate、AgentExecution 等持久化模型。
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

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
    description: str | None = Field(default=None, max_length=500, description="Agent 描述")
    agent_type: AgentType = Field(default=AgentType.SINGLE, description="Agent 类型")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Agent 状态")
    tenant_id: int | None = Field(default=None, description="租户 ID（业务关联）")
    created_by_user_id: int | None = Field(default=None, description="创建者用户 ID")
    is_builtin: bool = Field(default=False, description="是否内置 Agent")
    model_name: str = Field(default="gpt-4o-mini", max_length=50, description="使用的模型")
    system_prompt: str = Field(default="", description="系统提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int | None = Field(default=None, ge=1, description="最大生成 tokens")
    config: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB), description="额外配置"
    )


class Agent(AgentBase, table=True):
    """Agent 表模型"""

    __tablename__ = "agents"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentCreate(AgentBase):
    """Agent 创建模型"""


class AgentUpdate(SQLModel):
    """Agent 更新模型"""

    name: str | None = None
    description: str | None = None
    agent_type: AgentType | None = None
    status: AgentStatus | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    config: dict[str, Any] | None = None


class AgentPublic(AgentBase):
    """Agent 公开信息"""

    id: int
    created_at: datetime


# ============== PromptTemplate 模型 ==============


class PromptTemplateBase(SQLModel):
    """Prompt 模板基础模型"""

    name: str = Field(max_length=100, description="模板名称")
    description: str | None = Field(default=None, max_length=500, description="模板描述")
    category: str | None = Field(default=None, max_length=50, description="分类")
    template: str = Field(description="模板内容（支持 Jinja2）")
    variables: list[str] | None = Field(
        default=None, sa_column=Column(JSONB), description="变量列表"
    )
    is_active: bool = Field(default=True, description="是否启用")


class PromptTemplate(PromptTemplateBase, table=True):
    """Prompt 模板表模型"""

    __tablename__ = "prompt_templates"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PromptTemplateCreate(PromptTemplateBase):
    """Prompt 模板创建模型"""


class PromptTemplateUpdate(SQLModel):
    """Prompt 模板更新模型"""

    name: str | None = None
    description: str | None = None
    category: str | None = None
    template: str | None = None
    variables: list[str] | None = None
    is_active: bool | None = None


class PromptTemplatePublic(PromptTemplateBase):
    """Prompt 模板公开信息"""

    id: int
    created_at: datetime


# ============== MCPService 模型 ==============


class MCPServiceBase(SQLModel):
    """MCP 服务基础模型"""

    name: str = Field(max_length=255, description="服务名称")
    description: str | None = Field(default=None, description="服务描述")
    tenant_id: int | None = Field(default=None, description="租户 ID（业务关联）")
    enabled: bool = Field(default=True, description="是否启用")
    transport_type: str = Field(default="stdio", max_length=50, description="传输方式：stdio/http/sse")
    url: str | None = Field(default=None, max_length=512, description="HTTP/SSE URL")
    headers: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    auth_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    advanced_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    stdio_config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    env_vars: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))


class MCPService(MCPServiceBase, table=True):
    """MCP 服务表模型"""

    __tablename__ = "mcp_services"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class MCPServiceCreate(MCPServiceBase):
    """MCP 服务创建模型"""


class MCPServiceUpdate(SQLModel):
    """MCP 服务更新模型"""

    name: str | None = None
    description: str | None = None
    tenant_id: int | None = None
    enabled: bool | None = None
    transport_type: str | None = None
    url: str | None = None
    headers: dict[str, Any] | None = None
    auth_config: dict[str, Any] | None = None
    advanced_config: dict[str, Any] | None = None
    stdio_config: dict[str, Any] | None = None
    env_vars: dict[str, Any] | None = None


class MCPServicePublic(MCPServiceBase):
    """MCP 服务公开信息"""

    id: int
    created_at: datetime

# ============== AgentExecution 模型 ==============


class AgentExecutionBase(SQLModel):
    """Agent 执行基础模型"""

    thread_id: str = Field(max_length=255, description="线程 ID")
    agent_id: int = Field(description="Agent ID")
    input_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB), description="输入数据"
    )
    output_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSONB), description="输出数据"
    )
    status: str = Field(default="running", max_length=50, description="状态：running/success/error")
    error_message: str | None = Field(default=None, description="错误信息")
    tokens_used: int | None = Field(default=None, description="使用的 tokens")
    duration_ms: int | None = Field(default=None, description="执行耗时（毫秒）")
    extra_data: dict[str, Any] | None = Field(
        default=None, sa_column=Column("metadata", JSONB), description="额外元数据"
    )


class AgentExecution(AgentExecutionBase, table=True):
    """Agent 执行历史表模型"""

    __tablename__ = "agent_executions"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentExecutionCreate(AgentExecutionBase):
    """Agent 执行创建模型"""


class AgentExecutionPublic(AgentExecutionBase):
    """Agent 执行公开信息"""

    id: int
    created_at: datetime
