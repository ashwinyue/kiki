"""用户模型"""

from datetime import datetime

from sqlalchemy import DateTime
from sqlmodel import Column, Field, SQLModel


class UserBase(SQLModel):
    """用户基础模型"""

    username: str = Field(index=True, unique=True, max_length=50)
    email: str | None = Field(default=None, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)


class User(UserBase, table=True):
    """用户表模型"""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(), default=datetime.utcnow),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow),
    )


class UserCreate(SQLModel):
    """用户创建模型"""

    username: str
    email: str | None = None
    password: str


class UserRead(SQLModel):
    """用户读取模型"""

    id: int
    username: str
    email: str | None = None
    is_active: bool
    created_at: datetime


class UserUpdate(SQLModel):
    """用户更新模型"""

    email: str | None = None
    password: str | None = None
