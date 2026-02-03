"""用户模型

对齐 WeKnora99 表结构
"""

from typing import TYPE_CHECKING
from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel

from app.models.timestamp import TimestampMixin

if TYPE_CHECKING:
    from app.models.thread import Thread


def hash_password(password: str) -> str:
    """哈希密码（占位，需要 bcrypt）"""
    import hashlib

    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """验证密码（占位）"""

    return hash_password(password) == hashed


class UserBase(SQLModel):
    """用户基础模型"""

    username: str = Field(max_length=100)
    email: str = Field(max_length=255)
    avatar: str | None = Field(default=None, max_length=500)
    is_active: bool = Field(default=True)
    can_access_all_tenants: bool = Field(default=False)


class User(TimestampMixin, UserBase, table=True):
    """用户表模型"""

    __tablename__ = "users"

    id: str = Field(default=None, primary_key=True, max_length=36)  # UUID
    password_hash: str = Field(max_length=255)
    tenant_id: int | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)

    # 关系
    threads: list["Thread"] = Relationship(back_populates="user")

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return verify_password(password, self.password_hash)

    def set_password(self, password: str) -> None:
        """设置密码（哈希）"""
        self.password_hash = hash_password(password)


class UserCreate(SQLModel):
    """用户创建模型"""

    username: str
    email: str
    password: str
    avatar: str | None = None
    tenant_id: int | None = None


class UserUpdate(SQLModel):
    """用户更新模型"""

    email: str | None = None
    password: str | None = None
    avatar: str | None = None
    is_active: bool | None = None


class UserPublic(UserBase):
    """用户公开信息"""

    id: str
    tenant_id: int | None
    created_at: datetime
