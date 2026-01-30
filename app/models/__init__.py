"""数据模型定义（SQLModel）"""

from app.models.database import (
    # User
    User,
    UserCreate,
    UserUpdate,
    UserPublic,
    # Session
    ChatSession as Session,
    SessionCreate,
    SessionPublic,
    # Thread
    Thread,
    ThreadCreate,
    ThreadPublic,
    # Message
    Message,
    MessageCreate,
    MessagePublic,
    # Memory (from database.py)
    Memory,
    MemoryCreate,
    MemoryUpdate,
    MemoryPublic,
    # Token
    Token,
    TokenPayload,
)

from app.models.agent import (
    # Agent
    Agent,
    AgentType,
    AgentStatus,
    AgentCreate,
    AgentUpdate,
    AgentPublic,
    # Tool
    Tool,
    ToolCreate,
    ToolUpdate,
    ToolPublic,
    # Agent-Tool 关联
    AgentTool,
    # PromptTemplate
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplatePublic,
    # AgentExecution
    AgentExecution,
    AgentExecutionCreate,
    AgentExecutionPublic,
)

__all__ = [
    # User
    "User",
    "UserCreate",
    "UserUpdate",
    "UserPublic",
    # Session
    "Session",
    "SessionCreate",
    "SessionPublic",
    # Thread
    "Thread",
    "ThreadCreate",
    "ThreadPublic",
    # Message
    "Message",
    "MessageCreate",
    "MessagePublic",
    # Memory
    "Memory",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryPublic",
    # Token
    "Token",
    "TokenPayload",
    # Agent
    "Agent",
    "AgentType",
    "AgentStatus",
    "AgentCreate",
    "AgentUpdate",
    "AgentPublic",
    # Tool
    "Tool",
    "ToolCreate",
    "ToolUpdate",
    "ToolPublic",
    "AgentTool",
    # PromptTemplate
    "PromptTemplate",
    "PromptTemplateCreate",
    "PromptTemplateUpdate",
    "PromptTemplatePublic",
    # AgentExecution
    "AgentExecution",
    "AgentExecutionCreate",
    "AgentExecutionPublic",
]
