"""消息相关模式

提供消息 API 的请求/响应模型。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """消息响应"""

    id: int = Field(..., description="消息 ID")
    session_id: str = Field(..., description="会话 ID")
    role: str = Field(..., description="角色：user/assistant/system/tool")
    content: str = Field(..., description="消息内容")
    is_completed: bool = Field(..., description="是否完成")
    request_id: str | None = Field(None, description="请求 ID")
    knowledge_references: dict[str, Any] | None = Field(None, description="知识引用")
    agent_steps: dict[str, Any] | None = Field(None, description="Agent 执行步骤")
    tool_calls: dict[str, Any] | None = Field(None, description="工具调用")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class MessageUpdate(BaseModel):
    """更新消息请求"""

    content: str = Field(..., min_length=1, description="新消息内容")


class MessageRegenerateRequest(BaseModel):
    """消息重新生成请求"""

    regenerate: bool = Field(False, description="是否重新生成后续消息")


class MessageListResponse(BaseModel):
    """消息列表响应"""

    items: list[MessageResponse] = Field(default_factory=list, description="消息列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")
    pages: int = Field(..., description="总页数")


class MessageSearchResponse(BaseModel):
    """消息搜索响应"""

    items: list[MessageResponse] = Field(default_factory=list, description="匹配的消息列表")
    total: int = Field(..., description="总数")
    query: str = Field(..., description="搜索关键词")
