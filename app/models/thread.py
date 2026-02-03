"""线程模型

用于 LangGraph 状态持久化。
"""

from typing import TYPE_CHECKING
from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel

from app.models.timestamp import TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class ThreadBase(SQLModel):
    """线程基础模型"""

    name: str = Field(max_length=500)
    session_id: str | None = Field(default=None, max_length=255)


class Thread(TimestampMixin, ThreadBase, table=True):
    """线程表模型（用于 LangGraph 状态持久化）"""

    __tablename__ = "threads"

    # 主键（字符串类型的 thread_id）
    id: str = Field(max_length=255, primary_key=True)
    # 关联字段
    user_id: int | None = Field(default=None, foreign_key="users.id")
    tenant_id: int | None = Field(default=None)
    status: str = Field(default="active", max_length=50)  # active, archived, deleted
    deleted_at: datetime | None = Field(default=None)

    # 关系
    user: "User | None" = Relationship(back_populates="threads")
    # tenant: Tenant | None = Relationship(back_populates="threads")


class ThreadCreate(ThreadBase):
    """线程创建模型"""


class ThreadPublic(ThreadBase):
    """线程公开信息"""

    id: str
    user_id: int | None
    tenant_id: int | None
    status: str
    created_at: datetime


__all__ = [
    "Thread",
    "ThreadBase",
    "ThreadCreate",
    "ThreadPublic",
]
