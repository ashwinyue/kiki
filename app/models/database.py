"""数据库模型模块

使用 SQLModel 定义数据模型。
"""

from typing import TYPE_CHECKING, List, Any
from datetime import datetime, UTC

from sqlmodel import Field, Relationship, SQLModel, Column
from sqlalchemy import JSON
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============== 通用字段定义 ==============

def get_base_fields() -> dict:
    """获取通用基础字段

    由于 SQLModel 与 Pydantic v2 的兼容性问题，
    不使用基类继承，而是在各模型中直接添加字段。
    """
    return {
        "id": Field(default=None, primary_key=True),
        "created_at": Field(default_factory=lambda: datetime.now(UTC)),
        "updated_at": Field(default_factory=lambda: datetime.now(UTC)),
    }


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

    # 关系
    sessions: List["ChatSession"] = Relationship(back_populates="user")
    threads: List["Thread"] = Relationship(back_populates="user")

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(password, self.hashed_password)

    def set_password(self, password: str) -> None:
        """设置密码（哈希）"""
        self.hashed_password = pwd_context.hash(password)


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(min_length=8, max_length=100)


class UserUpdate(SQLModel):
    """用户更新模型"""
    full_name: str | None = None
    email: str | None = None
    password: str | None = None


class UserPublic(UserBase):
    """用户公开信息（不含密码）"""
    id: int


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
    extra_data: Any | None = Field(default=None, sa_column=Column(JSON))

    # 关系
    user: User | None = Relationship(back_populates="sessions")
    messages: List["Message"] = Relationship(back_populates="session")


# 向后兼容别名
Session = ChatSession


class SessionCreate(ChatSessionBase):
    """会话创建模型"""
    user_id: int | None = None


class SessionPublic(ChatSessionBase):
    """会话公开信息"""
    id: str
    user_id: int | None
    created_at: datetime
    message_count: int = 0


# ============== 线程模型 ==============

class ThreadBase(SQLModel):
    """线程基础模型"""
    name: str = Field(max_length=500)


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
    status: str = Field(default="active", max_length=50)  # active, archived, deleted

    # 关系
    user: User | None = Relationship(back_populates="threads")


class ThreadCreate(ThreadBase):
    """线程创建模型"""


class ThreadPublic(ThreadBase):
    """线程公开信息"""
    id: str
    user_id: int | None
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 关联字段
    session_id: str | None = Field(default=None, foreign_key="chatsessions.id")
    tool_calls: Any | None = Field(default=None, sa_column=Column(JSON))
    extra_data: Any | None = Field(default=None, sa_column=Column(JSON))

    # 关系
    session: ChatSession | None = Relationship(back_populates="messages")


class MessageCreate(MessageBase):
    """消息创建模型"""
    session_id: str | None = None


class MessagePublic(MessageBase):
    """消息公开信息"""
    id: int
    session_id: str | None
    created_at: datetime


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
    value: Any | None = Field(default=None, sa_column=Column(JSON))
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
