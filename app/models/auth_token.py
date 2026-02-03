"""认证令牌模型

对齐 WeKnora99 表结构
"""

from datetime import datetime

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Column, Field, SQLModel

from app.models.timestamp import TimestampMixin


class AuthTokenBase(SQLModel):
    """认证令牌基础模型"""

    user_id: str = Field(max_length=36)
    token: str
    token_type: str = Field(max_length=50)  # access_token, refresh_token, api_key
    expires_at: datetime
    is_revoked: bool = Field(default=False)
    name: str | None = Field(default=None, max_length=100)


class AuthToken(TimestampMixin, AuthTokenBase, table=True):
    """认证令牌表模型"""

    __tablename__ = "auth_tokens"

    id: str = Field(default=None, primary_key=True, max_length=36)
    scopes: list[str] | None = Field(
        default=None, sa_column=Column(ARRAY(String), default=None)
    )
    last_used_at: datetime | None = Field(default=None)


class AuthTokenCreate(SQLModel):
    """认证令牌创建模型"""

    user_id: str
    token: str
    token_type: str
    expires_at: datetime
    name: str | None = None
    scopes: list[str] | None = None


class AuthTokenPublic(SQLModel):
    """认证令牌公开信息"""

    id: str
    user_id: str
    token_type: str
    expires_at: datetime
    is_revoked: bool
    created_at: datetime
