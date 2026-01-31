"""数据库模型模块

向后兼容的导出层，从各独立模型文件重新导出。

已拆分的文件：
- app/models/user.py - 用户相关
- app/models/tenant.py - 租户相关
- app/models/session.py - 会话相关
- app/models/message.py - 消息相关
- app/models/memory.py - 长期记忆相关
"""

# ============== 重新导出各模型 ==============

# 用户
from app.models.user import (
    User,
    UserCreate,
    UserPublic,
    UserUpdate,
    hash_password,
    verify_password,
)

# 租户
from app.models.tenant import (
    Tenant,
    TenantCreate,
    TenantPublic,
    TenantUpdate,
)

# 会话
from app.models.session import (
    ChatSession,
    Session,
    SessionCreate,
    SessionPublic,
    SessionUpdate,
)

# 消息
from app.models.message import (
    Message,
    MessageCreate,
    MessagePublic,
    MessageUpdate,
)

# 线程
from app.models.thread import (
    Thread,
    ThreadCreate,
    ThreadPublic,
)

# 长期记忆
from app.models.memory import (
    Memory,
    MemoryCreate,
    MemoryPublic,
    MemoryUpdate,
)

# 任务
from app.models.task import (
    Task,
    TaskCreate,
    TaskUpdate,
    TaskPublic,
    TaskList,
    TaskLog,
    TaskLogCreate,
    TaskLogPublic,
)

# Token
from datetime import UTC, datetime

from sqlmodel import SQLModel


class Token(SQLModel):
    """Token 响应模型"""

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime | None = None
    user: UserPublic | None = None


class TokenPayload(SQLModel):
    """Token Payload"""

    sub: str | int  # user_id
    exp: int | None = None
    iat: int | None = None


__all__ = [
    # 用户
    "User",
    "UserCreate",
    "UserUpdate",
    "UserPublic",
    "hash_password",
    "verify_password",
    # 租户
    "Tenant",
    "TenantCreate",
    "TenantUpdate",
    "TenantPublic",
    # 会话
    "ChatSession",
    "Session",
    "SessionCreate",
    "SessionUpdate",
    "SessionPublic",
    # 消息
    "Message",
    "MessageCreate",
    "MessageUpdate",
    "MessagePublic",
    # 线程
    "Thread",
    "ThreadCreate",
    "ThreadPublic",
    # 长期记忆
    "Memory",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryPublic",
    # 任务
    "Task",
    "TaskCreate",
    "TaskUpdate",
    "TaskPublic",
    "TaskList",
    "TaskLog",
    "TaskLogCreate",
    "TaskLogPublic",
    # Token
    "Token",
    "TokenPayload",
]
