"""Pydantic 模式定义"""

# 统一响应
from app.schemas.response import (
    ApiResponse,
    DataResponse,
    PaginatedResponse,
    PaginationMeta,
)

# 聊天
from app.schemas.chat import (
    ChatRequest,
    StreamChatRequest,
    ChatResponse,
    Message,
    ChatHistoryResponse,
    ContextStatsResponse,
    SSEEvent,
)

# Agent
from app.schemas.agent import (
    AgentConfig,
    RouterAgentRequest,
    SupervisorAgentRequest,
    SwarmAgentRequest,
    MultiAgentChatRequest,
    MultiAgentChatResponse,
    AgentMessage,
    MultiAgentChatHistoryResponse,
    AgentSystemResponse,
    AgentRequest,
    AgentPublic,
    AgentDetailResponse,
    AgentListResponse,
    AgentStatsResponse,
    ExecutionItem,
    ExecutionListResponse,
)

# 租户
from app.schemas.tenant import (
    TenantListResponse,
    ApiKeyResponse,
)

# MCP 服务
from app.schemas.mcp_service import (
    MCPServiceRequest,
    MCPServiceResponse,
    MCPServiceListResponse,
)

# 工具
from app.schemas.tool import (
    ToolInfo,
    ToolsListResponse,
)

# 会话
from app.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    GenerateTitleRequest,
)

# 消息
from app.schemas.message import (
    MessageResponse,
    MessageUpdate,
    MessageRegenerateRequest,
    MessageListResponse,
    MessageSearchResponse,
)

# 评估
from app.schemas.evaluation import (
    RunEvaluationRequest,
    RunEvaluationStreamRequest,
    DatasetListItem,
    EvaluationRunResponse,
    EvaluationStatusResponse,
)

__all__ = [
    # 统一响应
    "ApiResponse",
    "DataResponse",
    "PaginatedResponse",
    "PaginationMeta",
    # 聊天
    "ChatRequest",
    "StreamChatRequest",
    "ChatResponse",
    "Message",
    "ChatHistoryResponse",
    "ContextStatsResponse",
    "SSEEvent",
    # Agent - 多 Agent 协作
    "AgentConfig",
    "RouterAgentRequest",
    "SupervisorAgentRequest",
    "SwarmAgentRequest",
    "MultiAgentChatRequest",
    "MultiAgentChatResponse",
    "AgentMessage",
    "MultiAgentChatHistoryResponse",
    "AgentSystemResponse",
    # Agent - CRUD
    "AgentRequest",
    "AgentPublic",
    "AgentDetailResponse",
    "AgentListResponse",
    "AgentStatsResponse",
    "ExecutionItem",
    "ExecutionListResponse",
    # 租户
    "TenantListResponse",
    "ApiKeyResponse",
    # MCP 服务
    "MCPServiceRequest",
    "MCPServiceResponse",
    "MCPServiceListResponse",
    # 工具
    "ToolInfo",
    "ToolsListResponse",
    # 会话
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionDetailResponse",
    "SessionListResponse",
    "GenerateTitleRequest",
    # 消息
    "MessageResponse",
    "MessageUpdate",
    "MessageRegenerateRequest",
    "MessageListResponse",
    "MessageSearchResponse",
    # 评估
    "RunEvaluationRequest",
    "RunEvaluationStreamRequest",
    "DatasetListItem",
    "EvaluationRunResponse",
    "EvaluationStatusResponse",
]
