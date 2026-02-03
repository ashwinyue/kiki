"""Agent 相关模式

提供单 Agent CRUD 相关的请求/响应模型。
"""

from typing import Any

from pydantic import BaseModel, Field

# ============== Agent 配置 Schemas ==============


class AgentConfig(BaseModel):
    """Agent 基础配置"""

    name: str = Field(..., description="Agent 名称")
    system_prompt: str = Field("", description="系统提示词")
    tools: list[str] = Field(default_factory=list, description="使用的工具名称列表")


# ============== 单 Agent CRUD Schemas ==============


class AgentRequest(BaseModel):
    """Agent 创建/更新请求（对齐 WeKnora CustomAgent）"""

    name: str = Field(..., description="Agent 名称", min_length=1, max_length=255)
    description: str | None = Field(None, description="Agent 描述", max_length=500)
    config: dict = Field(default_factory=dict, description="Agent 配置（JSON 格式）")


class AgentPublic(BaseModel):
    """Agent 公开信息"""

    id: str
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

    id: str
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


# ============== Agent 复制 Schemas ==============


class AgentCopyRequest(BaseModel):
    """Agent 复制请求（对齐 WeKnora Copy API）"""

    name: str = Field(..., description="新 Agent 名称", min_length=1, max_length=255)
    description: str | None = Field(None, description="新 Agent 描述", max_length=500)
    copy_config: bool = Field(default=True, description="是否复制配置")
    copy_tools: bool = Field(default=True, description="是否复制工具关联")
    copy_knowledge: bool = Field(default=False, description="是否复制知识库关联")


class AgentCopyResponse(BaseModel):
    """Agent 复制响应"""

    source_agent_id: str = Field(..., description="源 Agent ID")
    new_agent_id: str = Field(..., description="新 Agent ID")
    name: str = Field(..., description="新 Agent 名称")
    description: str | None = Field(None, description="新 Agent 描述")
    config: dict = Field(default_factory=dict, description="复制的配置")
    copied_tools: list[str] = Field(default_factory=list, description="复制的工具 ID 列表")
    copied_knowledge: list[str] = Field(
        default_factory=list, description="复制的知识库 ID 列表"
    )


class BatchAgentCopyRequest(BaseModel):
    """批量 Agent 复制请求"""

    agent_ids: list[str] = Field(..., description="要复制的 Agent ID 列表")
    name_suffix: str = Field(
        default=" (副本)", description="名称后缀，添加到原名称后"
    )
    copy_config: bool = Field(default=True, description="是否复制配置")
    copy_tools: bool = Field(default=True, description="是否复制工具关联")
    copy_knowledge: bool = Field(default=False, description="是否复制知识库关联")


class BatchAgentCopyResponse(BaseModel):
    """批量 Agent 复制响应"""

    success_count: int = Field(..., description="成功复制数量")
    failed_count: int = Field(..., description="失败数量")
    results: list[AgentCopyResponse] = Field(
        default_factory=list, description="复制结果列表"
    )
    errors: dict[str, str] = Field(
        default_factory=dict, description="失败的 Agent ID 及错误信息"
    )


# ============== 占位符 Schemas ==============


class PlaceholderSchema(BaseModel):
    """占位符模式"""

    id: str
    name: str
    description: str | None
    default_value: str | None
    variable_type: str
    validation_rule: str | None
    is_required: bool
    is_enabled: bool
    agent_id: str | None
    category: str | None
    display_order: int
    created_at: str


class PlaceholderCreate(BaseModel):
    """创建占位符请求"""

    name: str = Field(..., description="变量名", min_length=1, max_length=100)
    description: str | None = Field(None, description="变量描述", max_length=500)
    default_value: str | None = Field(None, description="默认值", max_length=2000)
    variable_type: str = Field(
        default="string",
        description="变量类型: string, number, boolean, json, array",
    )
    validation_rule: str | None = Field(None, description="验证规则", max_length=500)
    is_required: bool = Field(default=False, description="是否必填")
    is_enabled: bool = Field(default=True, description="是否启用")
    agent_id: str | None = Field(None, description="关联的 Agent ID")
    category: str | None = Field(None, description="分类")
    display_order: int = Field(default=0, description="显示顺序")


class PlaceholderUpdate(BaseModel):
    """更新占位符请求"""

    name: str | None = Field(None, description="变量名", max_length=100)
    description: str | None = Field(None, description="变量描述", max_length=500)
    default_value: str | None = Field(None, description="默认值", max_length=2000)
    variable_type: str | None = Field(None, description="变量类型")
    validation_rule: str | None = Field(None, description="验证规则", max_length=500)
    is_required: bool | None = Field(None, description="是否必填")
    is_enabled: bool | None = Field(None, description="是否启用")
    category: str | None = Field(None, description="分类")
    display_order: int | None = Field(None, description="显示顺序")


class PlaceholderListResponse(BaseModel):
    """占位符列表响应"""

    items: list[PlaceholderSchema]
    total: int


class PlaceholderPreviewRequest(BaseModel):
    """预览占位符替换请求"""

    template: str = Field(..., description="模板内容（包含占位符）")
    values: dict[str, Any] = Field(default_factory=dict, description="变量值")
    agent_id: str | None = Field(None, description="Agent ID（用于获取关联占位符）")


class PlaceholderPreviewResponse(BaseModel):
    """预览占位符替换响应"""

    rendered: str = Field(..., description="渲染后的内容")
    missing_variables: list[str] = Field(
        default_factory=list, description="缺失的变量"
    )
    validation_errors: dict[str, str] = Field(
        default_factory=dict, description="验证错误"
    )
    used_placeholders: list[str] = Field(
        default_factory=list, description="使用的占位符"
    )
