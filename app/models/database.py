"""数据库模型模块

使用 SQLModel 定义数据模型。
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import bcrypt
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Relationship, SQLModel


def hash_password(password: str) -> str:
    """哈希密码"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ============== 通用字段定义 ==============


def get_base_fields() -> dict:
    """获取通用基础字段

    由于 SQLModel 与 Pydantic v2 的兼容性问题，
    不使用基类继承，而是在各模型中直接添加字段。
    """
    def now_utc() -> datetime:
        return datetime.now(UTC)

    return {
        "id": Field(default=None, primary_key=True),
        "created_at": Field(default_factory=now_utc),
        "updated_at": Field(default_factory=now_utc),
    }


# ============== 租户模型 ==============


class TenantBase(SQLModel):
    """租户基础模型"""

    name: str = Field(max_length=255)
    description: str | None = Field(default=None)
    api_key: str = Field(max_length=64, index=True, unique=True)
    status: str = Field(default="active", max_length=50)
    config: Any | None = Field(default=None, sa_column=Column(JSONB))


class Tenant(TenantBase, table=True):
    """租户表模型"""

    __tablename__ = "tenants"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)

    # 关系
    users: list["User"] = Relationship(back_populates="tenant")
    sessions: list["ChatSession"] = Relationship(back_populates="tenant")
    threads: list["Thread"] = Relationship(back_populates="tenant")


class TenantCreate(SQLModel):
    """租户创建模型"""

    name: str = Field(max_length=255)
    description: str | None = Field(default=None)
    status: str = Field(default="active", max_length=50)
    config: Any | None = Field(default=None)


class TenantUpdate(SQLModel):
    """租户更新模型"""

    name: str | None = None
    description: str | None = None
    api_key: str | None = None
    status: str | None = None
    config: Any | None = None


class TenantPublic(TenantBase):
    """租户公开信息"""

    id: int
    created_at: datetime


# ============== 用户模型 ==============


class UserBase(SQLModel):
    """用户基础模型"""

    email: str = Field(unique=True, index=True, max_length=255)
    full_name: str | None = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)


class User(UserBase, table=True):
    """用户表模型"""

    __tablename__ = "users"

    # 基础字段（直接定义，避免继承问题）
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    hashed_password: str = Field(max_length=255)
    tenant_id: int | None = Field(default=None)
    can_access_all_tenants: bool = Field(default=False)

    # 关系
    tenant: Tenant | None = Relationship(back_populates="users")
    sessions: list["ChatSession"] = Relationship(back_populates="user")
    threads: list["Thread"] = Relationship(back_populates="user")

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return verify_password(password, self.hashed_password)

    def set_password(self, password: str) -> None:
        """设置密码（哈希）"""
        self.hashed_password = hash_password(password)


class UserCreate(UserBase):
    """用户创建模型"""

    password: str = Field(min_length=8, max_length=100)
    tenant_id: int | None = None
    can_access_all_tenants: bool = False


class UserUpdate(SQLModel):
    """用户更新模型"""

    full_name: str | None = None
    email: str | None = None
    password: str | None = None
    tenant_id: int | None = None
    can_access_all_tenants: bool | None = None


class UserPublic(UserBase):
    """用户公开信息（不含密码）"""

    id: int
    tenant_id: int | None = None
    can_access_all_tenants: bool = False


# ============== 会话模型 ==============


class ChatSessionBase(SQLModel):
    """会话基础模型"""

    name: str = Field(default="", max_length=500)


class ChatSession(ChatSessionBase, table=True):
    """会话表模型"""

    __tablename__ = "chatsessions"

    # 主键（字符串类型的 UUID）
    id: str = Field(max_length=255, primary_key=True)
    # 基础字段
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 关联字段
    user_id: int | None = Field(default=None, foreign_key="users.id")
    tenant_id: int | None = Field(default=None)
    agent_id: int | None = Field(default=None)
    agent_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    context_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    extra_data: Any | None = Field(default=None, sa_column=Column(JSONB))
    deleted_at: datetime | None = Field(default=None)

    # 关系
    user: User | None = Relationship(back_populates="sessions")
    tenant: Tenant | None = Relationship(back_populates="sessions")
    messages: list["Message"] = Relationship(back_populates="session")


# 向后兼容别名
Session = ChatSession


class SessionCreate(ChatSessionBase):
    """会话创建模型"""

    user_id: int | None = None
    tenant_id: int | None = None
    agent_id: int | None = None
    agent_config: Any | None = None
    context_config: Any | None = None


class SessionPublic(ChatSessionBase):
    """会话公开信息"""

    id: str
    user_id: int | None
    tenant_id: int | None
    agent_id: int | None
    created_at: datetime
    message_count: int = 0


# ============== 线程模型 ==============


class ThreadBase(SQLModel):
    """线程基础模型"""

    name: str = Field(max_length=500)
    session_id: str | None = Field(default=None, max_length=255)


class Thread(ThreadBase, table=True):
    """线程表模型（用于 LangGraph 状态持久化）"""

    __tablename__ = "threads"

    # 主键（字符串类型的 thread_id）
    id: str = Field(max_length=255, primary_key=True)
    # 基础字段
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 关联字段
    user_id: int | None = Field(default=None, foreign_key="users.id")
    tenant_id: int | None = Field(default=None)
    status: str = Field(default="active", max_length=50)  # active, archived, deleted
    deleted_at: datetime | None = Field(default=None)

    # 关系
    user: User | None = Relationship(back_populates="threads")
    tenant: Tenant | None = Relationship(back_populates="threads")


class ThreadCreate(ThreadBase):
    """线程创建模型"""


class ThreadPublic(ThreadBase):
    """线程公开信息"""

    id: str
    user_id: int | None
    tenant_id: int | None
    status: str
    created_at: datetime


# ============== 消息模型 ==============


class MessageBase(SQLModel):
    """消息基础模型"""

    role: str = Field(max_length=50)  # user, assistant, system, tool
    content: str = Field(default="")


class Message(MessageBase, table=True):
    """消息表模型"""

    __tablename__ = "messages"

    # 主键和基础字段
    id: int | None = Field(default=None, primary_key=True)
    request_id: str | None = Field(default=None, max_length=255)
    knowledge_references: Any | None = Field(default=None, sa_column=Column(JSONB))
    agent_steps: Any | None = Field(default=None, sa_column=Column(JSONB))
    is_completed: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)
    # 关联字段
    session_id: str | None = Field(default=None, foreign_key="chatsessions.id")
    tool_calls: Any | None = Field(default=None, sa_column=Column(JSONB))
    extra_data: Any | None = Field(default=None, sa_column=Column(JSONB))

    # 关系
    session: ChatSession | None = Relationship(back_populates="messages")


class MessageCreate(MessageBase):
    """消息创建模型"""

    session_id: str | None = None
    request_id: str | None = None
    knowledge_references: Any | None = None
    agent_steps: Any | None = None
    is_completed: bool | None = None


class MessagePublic(MessageBase):
    """消息公开信息"""

    id: int
    session_id: str | None
    created_at: datetime
    request_id: str | None
    is_completed: bool


# ============== 长期记忆模型 ==============


class MemoryBase(SQLModel):
    """长期记忆基础模型"""

    namespace: str = Field(max_length=255, index=True)
    key: str = Field(max_length=500)


class Memory(MemoryBase, table=True):
    """长期记忆表模型（LangGraph Store 持久化）

    用于存储跨会话的长期记忆，如用户偏好、对话摘要等。
    """

    __tablename__ = "memories"

    # 联合主键
    namespace: str = Field(max_length=255, primary_key=True)
    key: str = Field(max_length=500, primary_key=True)
    # 基础字段
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 存储值（JSONB）
    value: Any | None = Field(default=None, sa_column=Column(JSONB))
    # 过期时间（可选）
    expires_at: datetime | None = Field(default=None)


class MemoryCreate(MemoryBase):
    """记忆创建模型"""

    value: Any


class MemoryUpdate(SQLModel):
    """记忆更新模型"""

    value: Any


class MemoryPublic(MemoryBase):
    """记忆公开信息"""

    value: Any
    created_at: datetime
    updated_at: datetime


# ============== Token 模型 ==============


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


# 解决循环导入
if TYPE_CHECKING:
    pass
