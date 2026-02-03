"""数据模型定义（SQLModel）

Multi-Agent 架构支持，对齐 LangGraph 最佳实践。
"""

# ============== 基础 Mixin ==============
from app.models.timestamp import TimestampMixin

# ============== 用户认证 ==============
from app.models.auth_token import (
    AuthToken,
    AuthTokenCreate,
    AuthTokenPublic,
)

# ============== Agent ==============
from app.models.agent_execution import (
    AgentExecution,
    AgentExecutionBase,
    AgentExecutionCreate,
    AgentExecutionPublic,
    AgentExecutionUpdate,
    AgentType,
    ExecutionStatus,
)
from app.models.agent_config import (
    # 新名称（推荐使用）
    AgentConfig,
    AgentConfigBase,
    AgentConfigCreate,
    AgentConfigUpdate,
    AgentConfigPublic,
    # 旧名称（向后兼容，已废弃）
    Agent,
    AgentCreate,
    AgentPublic,
    AgentRole,
    AgentUpdate,
    CustomAgent,
    CustomAgentCreate,
    CustomAgentPublic,
    CustomAgentUpdate,
    GraphType,
)

# ============== LLM 模型 ==============
from app.models.llm_model import (
    Model,
    ModelBase,
    ModelCreate,
    ModelUpdate,
)
)

# ============== MCP 服务 ==============
from app.models.mcp_service import (
    MCPService,
    MCPServiceCreate,
    MCPServicePublic,
    MCPServiceUpdate,
)

# ============== 会话 ==============
from app.models.memory import (
    Memory,
    MemoryCreate,
    MemoryPublic,
    MemoryUpdate,
)

# ============== 消息 ==============
from app.models.message import (
    Message,
    MessageCreate,
    MessagePublic,
    MessageUpdate,
)
from app.models.session import (
    ChatSession,
    Session,
    SessionCreate,
    SessionPublic,
    SessionUpdate,
)

# ============== 租户 ==============
from app.models.tenant import (
    Tenant,
    TenantCreate,
    TenantPublic,
    TenantUpdate,
)
from app.models.thread import (
    Thread,
    ThreadCreate,
    ThreadPublic,
)
from app.models.timestamp import TimestampMixin
from app.models.user import (
    User,
    UserCreate,
    UserPublic,
    UserUpdate,
    hash_password,
    verify_password,
)

__all__ = [
    # 基础 Mixin
    "TimestampMixin",
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
    # Agent Config
    "AgentConfig",
    "AgentConfigBase",
    "AgentConfigCreate",
    "AgentConfigUpdate",
    "AgentConfigPublic",
    # 向后兼容（已废弃）
    "Agent",
    "AgentCreate",
    "AgentUpdate",
    "AgentPublic",
    "AgentRole",
    "GraphType",
    "CustomAgent",
    "CustomAgentCreate",
    "CustomAgentUpdate",
    "CustomAgentPublic",
    # Agent Execution
    "AgentExecution",
    "AgentExecutionBase",
    "AgentExecutionCreate",
    "AgentExecutionUpdate",
    "AgentExecutionPublic",
    "AgentType",
    "ExecutionStatus",
    # MCP 服务
    "MCPService",
    "MCPServiceCreate",
    "MCPServiceUpdate",
    "MCPServicePublic",
]
