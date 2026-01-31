"""会话相关模式

提供会话 API 的请求/响应模型。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """创建会话请求"""

    name: str = Field(..., min_length=1, max_length=500, description="会话名称")
    agent_id: int | None = Field(None, description="关联的 Agent ID")
    agent_config: dict[str, Any] | None = Field(None, description="Agent 配置")
    context_config: dict[str, Any] | None = Field(None, description="上下文配置")
    extra_data: dict[str, Any] | None = Field(None, description="额外数据")


class SessionUpdate(BaseModel):
    """更新会话请求"""

    name: str | None = Field(None, min_length=1, max_length=500, description="会话名称")
    agent_id: int | None = Field(None, description="关联的 Agent ID")
    agent_config: dict[str, Any] | None = Field(None, description="Agent 配置")
    context_config: dict[str, Any] | None = Field(None, description="上下文配置")
    extra_data: dict[str, Any] | None = Field(None, description="额外数据")


class SessionResponse(BaseModel):
    """会话响应"""

    id: str = Field(..., description="会话 ID")
    name: str = Field(..., description="会话名称")
    user_id: int | None = Field(None, description="用户 ID")
    tenant_id: int | None = Field(None, description="租户 ID")
    agent_id: int | None = Field(None, description="关联的 Agent ID")
    message_count: int = Field(..., description="消息数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class SessionDetailResponse(SessionResponse):
    """会话详情响应"""

    agent_config: dict[str, Any] | None = Field(None, description="Agent 配置")
    context_config: dict[str, Any] | None = Field(None, description="上下文配置")
    extra_data: dict[str, Any] | None = Field(None, description="额外数据")


class SessionListResponse(BaseModel):
    """会话列表响应"""

    items: list[SessionResponse] = Field(default_factory=list, description="会话列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")
    pages: int = Field(..., description="总页数")


class GenerateTitleRequest(BaseModel):
    """生成标题请求"""

    model_name: str | None = Field(None, description="使用的模型名称，为空则使用默认模型")
