"""聊天相关模式

提供聊天 API 的请求/响应模型。
"""

from typing import Any

from pydantic import BaseModel, Field


class WebsearchConfig(BaseModel):
    """Web 搜索配置

    类似 WeKnora 的 websearch 选择接口。
    """

    enable: bool = Field(False, description="是否启用 Web 搜索")
    provider: str = Field(
        "auto",
        description="搜索引擎: auto(自动选择), duckduckgo, tavily",
    )
    search_depth: str = Field(
        "basic",
        description="搜索深度: basic(基础), advanced(深度，仅 Tavily 支持)",
    )
    max_results: int = Field(5, ge=1, le=10, description="最大结果数")


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str = Field(..., description="会话 ID")
    user_id: str | None = Field(None, description="用户 ID")
    websearch: WebsearchConfig | None = Field(None, description="Web 搜索配置")


class StreamChatRequest(ChatRequest):
    """流式聊天请求"""

    stream_mode: str = Field(
        "messages",
        description="流式模式: messages(令牌级), updates(状态更新), values(完整状态)",
    )


class ChatResponse(BaseModel):
    """聊天响应"""

    content: str = Field(..., description="响应内容")
    session_id: str = Field(..., description="会话 ID")


class Message(BaseModel):
    """消息"""

    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""

    messages: list[Message] = Field(default_factory=list, description="历史消息")
    session_id: str = Field(..., description="会话 ID")


class ContextStatsResponse(BaseModel):
    """上下文统计响应"""

    session_id: str
    message_count: int
    token_estimate: int
    role_distribution: dict[str, int]
    exists: bool


class SSEEvent(BaseModel):
    """SSE 事件模型"""

    event: str = Field(default="message", description="事件类型")
    data: dict[str, Any] = Field(..., description="事件数据")

    def format(self) -> str:
        """格式化为 SSE 格式

        Returns:
            SSE 格式字符串
        """
        import json

        data_str = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\\ndata: {data_str}\\n\\n"


class SearchProviderInfo(BaseModel):
    """搜索提供商信息"""

    name: str = Field(..., description="提供商名称")
    display_name: str = Field(..., description="显示名称")
    available: bool = Field(..., description="是否可用")
    requires_api_key: bool = Field(False, description="是否需要 API Key")
    supported_depths: list[str] = Field(default_factory=list, description="支持的搜索深度")
    description: str = Field(..., description="描述")


class SearchProvidersResponse(BaseModel):
    """搜索提供商列表响应"""

    providers: list[SearchProviderInfo] = Field(default_factory=list, description="提供商列表")
    default_provider: str = Field(..., description="默认提供商")
