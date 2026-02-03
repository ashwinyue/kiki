"""数据库模型模块

向后兼容的导出层，从各独立模型文件重新导出。

已拆分的文件：
- app/models/user.py - 用户相关
- app/models/tenant.py - 租户相关
- app/models/session.py - 会话相关
- app/models/message.py - 消息相关
- app/models/memory.py - 长期记忆相关
"""



from datetime import datetime

from sqlmodel import SQLModel

from app.models.memory import (
    Memory,
    MemoryCreate,
    MemoryPublic,
    MemoryUpdate,
)

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
from app.models.user import (
    User,
    UserCreate,
    UserPublic,
    UserUpdate,
    hash_password,
    verify_password,
)


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
    "User",
    "UserCreate",
    "UserUpdate",
    "UserPublic",
    "hash_password",
    "verify_password",
    "Tenant",
    "TenantCreate",
    "TenantUpdate",
    "TenantPublic",
    "ChatSession",
    "Session",
    "SessionCreate",
    "SessionUpdate",
    "SessionPublic",
    "Message",
    "MessageCreate",
    "MessageUpdate",
    "MessagePublic",
    "Thread",
    "ThreadCreate",
    "ThreadPublic",
    "Memory",
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryPublic",
    "Token",
    "TokenPayload",
]
