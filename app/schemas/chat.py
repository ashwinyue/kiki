"""聊天相关模式"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: str | None = Field(None, description="会话 ID（续聊时提供）")
    stream: bool = Field(False, description="是否流式返回")


class ChatResponse(BaseModel):
    """聊天响应"""

    message: str = Field(..., description="AI 回复")
    session_id: str = Field(..., description="会话 ID")
    finished: bool = Field(True, description="是否完成")


class Message(BaseModel):
    """消息"""

    role: str = Field(..., description="角色 (user/assistant/system)")
    content: str = Field(..., description="消息内容")


class ChatHistory(BaseModel):
    """聊天历史"""

    messages: list[Message] = Field(default_factory=list, description="消息列表")
    session_id: str = Field(..., description="会话 ID")
