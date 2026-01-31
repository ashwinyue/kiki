"""多 Agent API 请求/响应模型"""

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent 基础配置"""

    name: str = Field(..., description="Agent 名称")
    system_prompt: str = Field("", description="系统提示词")
    tools: list[str] = Field(default_factory=list, description="使用的工具名称列表")


class RouterAgentRequest(BaseModel):
    """路由 Agent 创建请求"""

    name: str = Field(..., description="Agent 系统名称")
    agents: list[AgentConfig] = Field(..., description="子 Agent 配置列表", min_length=1)
    router_prompt: str | None = Field(None, description="自定义路由提示词")


class SupervisorAgentRequest(BaseModel):
    """监督 Agent 创建请求"""

    name: str = Field(..., description="Agent 系统名称")
    workers: list[AgentConfig] = Field(..., description="Worker Agent 配置列表", min_length=1)
    supervisor_prompt: str | None = Field(None, description="自定义监督提示词")


class SwarmAgentRequest(BaseModel):
    """Swarm Agent 创建请求"""

    name: str = Field(..., description="Agent 系统名称")
    agents: list[AgentConfig] = Field(
        ...,
        description="Agent 配置列表，每个 agent 可指定可切换的目标",
        min_length=1,
    )
    handoff_mapping: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Agent 切换映射 {agent_name: [可切换的目标列表]}",
    )
    default_agent: str = Field(
        ...,
        description="默认激活的 Agent 名称",
    )


class ChatRequest(BaseModel):
    """多 Agent 聊天请求"""

    message: str = Field(..., description="用户消息", min_length=1)
    session_id: str = Field(..., description="会话 ID")
    user_id: str | None = Field(None, description="用户 ID")
    stream: bool = Field(False, description="是否使用流式响应")


class ChatResponse(BaseModel):
    """聊天响应"""

    content: str = Field(..., description="响应内容")
    session_id: str = Field(..., description="会话 ID")
    agent_name: str | None = Field(None, description="最后响应的 Agent 名称")


class Message(BaseModel):
    """消息"""

    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""

    messages: list[Message] = Field(default_factory=list, description="历史消息")
    session_id: str = Field(..., description="会话 ID")


class AgentSystemResponse(BaseModel):
    """Agent 系统创建响应"""

    name: str = Field(..., description="系统名称")
    type: str = Field(..., description="系统类型: router/supervisor/swarm")
    agents: list[str] = Field(..., description="包含的 Agent 名称列表")
    session_id: str = Field(..., description="会话 ID")


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
