"""Agent 相关模式

提供单 Agent CRUD 相关的请求/响应模型。
"""

from pydantic import BaseModel, Field


# ============== Agent 配置 Schemas ==============


class AgentConfig(BaseModel):
    """Agent 基础配置"""

    name: str = Field(..., description="Agent 名称")
    system_prompt: str = Field("", description="系统提示词")
    tools: list[str] = Field(default_factory=list, description="使用的工具名称列表")


# ============== 单 Agent CRUD Schemas ==============


class AgentRequest(BaseModel):
    """Agent 创建/更新请求"""

    name: str = Field(..., description="Agent 名称", min_length=1, max_length=100)
    description: str | None = Field(None, description="Agent 描述", max_length=500)
    agent_type: str = Field("single", description="Agent 类型")
    model_name: str = Field("gpt-4o-mini", description="使用的模型")
    system_prompt: str | None = Field(None, description="系统提示词")
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    max_tokens: int = Field(0, description="最大生成 tokens", ge=0)
    config: dict = Field(default_factory=dict, description="额外配置")


class AgentPublic(BaseModel):
    """Agent 公开信息"""

    id: int
    name: str
    description: str | None
    agent_type: str
    status: str
    model_name: str
    system_prompt: str | None
    temperature: float
    max_tokens: int
    config: dict
    created_at: str


class AgentDetailResponse(BaseModel):
    """Agent 详情响应"""

    id: int
    name: str
    description: str | None
    agent_type: str
    status: str
    model_name: str
    system_prompt: str | None
    temperature: float
    max_tokens: int
    config: dict
    created_at: str


class AgentListResponse(BaseModel):
    """Agent 列表响应"""

    items: list[AgentPublic]
    total: int
    page: int
    size: int
    pages: int


class AgentStatsResponse(BaseModel):
    """Agent 统计响应"""

    total_agents: int
    active_agents: int
    agents_by_type: dict[str, int]


class ExecutionItem(BaseModel):
    """执行记录项"""

    id: int
    thread_id: str
    agent_id: int
    status: str
    tokens_used: int
    duration_ms: int
    created_at: str


class ExecutionListResponse(BaseModel):
    """执行历史响应"""

    items: list[ExecutionItem]
