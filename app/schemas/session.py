"""会话相关模式

提供会话 API 的请求/响应模型。
"""

from datetime import datetime
from enum import Enum
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


# ============== 会话状态相关模式 ==============


class SessionStateEnum(str, Enum):
    """会话状态枚举（用于 API 响应）"""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class SessionStopRequest(BaseModel):
    """停止会话请求"""

    reason: str | None = Field(None, description="停止原因")
    force: bool = Field(default=False, description="是否强制停止")


# 默认实例（用于 FastAPI 参数默认值）
_DEFAULT_SESSION_STOP_REQUEST = SessionStopRequest()


class SessionStopResponse(BaseModel):
    """停止会话响应"""

    session_id: str = Field(..., description="会话 ID")
    stopped: bool = Field(..., description="是否已停止")
    state: str = Field(..., description="当前状态")
    message: str = Field(..., description="响应消息")


class SessionStateResponse(BaseModel):
    """会话状态响应"""

    session_id: str = Field(..., description="会话 ID")
    state: SessionStateEnum = Field(..., description="会话状态")
    is_running: bool = Field(..., description="是否运行中")
    is_stopping: bool = Field(..., description="是否停止中")


# ============== 继续流相关模式 ==============


class ContinueStreamRequest(BaseModel):
    """继续流请求"""

    since: int = Field(
        default=0,
        ge=0,
        description="从第几个事件开始（0 表示从头开始）",
    )
    timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="等待超时时间（秒）",
    )


class ContinueStreamResponse(BaseModel):
    """继续流响应（SSE 流式返回）"""

    class Config:
        """Pydantic 配置"""

        # 这个模型仅用于文档，实际返回的是 SSE 流
        pass


class StreamInfoResponse(BaseModel):
    """流信息响应"""

    session_id: str = Field(..., description="会话 ID")
    is_active: bool = Field(..., description="流是否活跃")
    event_count: int = Field(..., description="事件数量")
    started_at: datetime | None = Field(None, description="开始时间")
    updated_at: datetime | None = Field(None, description="更新时间")
