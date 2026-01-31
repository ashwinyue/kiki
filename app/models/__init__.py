"""数据模型定义（SQLModel）

对齐 WeKnora99 表结构
"""

# ============== 用户认证 ==============
from app.models.user import (
    User,
    UserCreate,
    UserPublic,
    UserUpdate,
    hash_password,
    verify_password,
)
from app.models.auth_token import (
    AuthToken,
    AuthTokenCreate,
    AuthTokenPublic,
)

# ============== 租户 ==============
from app.models.tenant import (
    Tenant,
    TenantCreate,
    TenantPublic,
    TenantUpdate,
)

# ============== 会话 ==============
from app.models.memory import (
    Memory,
    MemoryCreate,
    MemoryPublic,
    MemoryUpdate,
)
from app.models.session import (
    ChatSession,
    Session,
    SessionCreate,
    SessionPublic,
    SessionUpdate,
)
from app.models.thread import (
    Thread,
    ThreadCreate,
    ThreadPublic,
)

# ============== 消息 ==============
from app.models.message import (
    Message,
    MessageCreate,
    MessagePublic,
    MessageUpdate,
)

# ============== 知识库 ==============
from app.models.knowledge import (
    # Model
    Model,
    ModelCreate,
    # KnowledgeBase
    KnowledgeBase,
    KnowledgeBaseCreate,
    # Knowledge
    Knowledge,
    KnowledgeCreate,
    # Chunk
    Chunk,
    ChunkCreate,
    # Embedding
    Embedding,
    EmbeddingBase,
    # KnowledgeTag
    KnowledgeTag,
    KnowledgeTagCreate,
)

# ============== Agent ==============
from app.models.custom_agent import (
    Agent,
    AgentCreate,
    AgentPublic,
    AgentUpdate,
    CustomAgent,
    CustomAgentCreate,
    CustomAgentPublic,
    CustomAgentUpdate,
)

# ============== MCP 服务 ==============
from app.models.mcp_service import (
    MCPService,
    MCPServiceCreate,
    MCPServicePublic,
    MCPServiceUpdate,
)

__all__ = [
    # 用户认证
    "User",
    "UserCreate",
    "UserUpdate",
    "UserPublic",
    "hash_password",
    "verify_password",
    "AuthToken",
    "AuthTokenCreate",
    "AuthTokenPublic",
    # 租户
    "Tenant",
    "TenantCreate",
    "TenantUpdate",
    "TenantPublic",
    # 会话
    "Session",
    "ChatSession",
    "SessionCreate",
    "SessionUpdate",
    "SessionPublic",
    # 线程
    "Thread",
    "ThreadCreate",
    "ThreadPublic",
    # 长期记忆
    "Memory",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryPublic",
    # 消息
    "Message",
    "MessageCreate",
    "MessageUpdate",
    "MessagePublic",
    # 知识库
    "Model",
    "ModelCreate",
    "KnowledgeBase",
    "KnowledgeBaseCreate",
    "Knowledge",
    "KnowledgeCreate",
    "Chunk",
    "ChunkCreate",
    "Embedding",
    "KnowledgeTag",
    "KnowledgeTagCreate",
    # Agent
    "Agent",
    "AgentCreate",
    "AgentUpdate",
    "AgentPublic",
    "CustomAgent",
    "CustomAgentCreate",
    "CustomAgentUpdate",
    "CustomAgentPublic",
    # MCP 服务
    "MCPService",
    "MCPServiceCreate",
    "MCPServiceUpdate",
    "MCPServicePublic",
]
