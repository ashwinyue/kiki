"""认证相关模式

提供用户注册、登录、Token 等相关的请求和响应模型。
包含完整的输入验证逻辑。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.schemas.response import ApiResponse

# ============== 请求模型 ==============


class RegisterRequest(BaseModel):
    """注册请求

    包含完整的字段验证：邮箱格式、密码复杂度、用户名规则。
    """

    email: EmailStr = Field(..., description="邮箱地址")
    password: str = Field(..., min_length=8, max_length=100, description="密码")
    full_name: str | None = Field(None, max_length=255, description="全名")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """验证密码复杂度

        规则：
        - 至少 8 个字符
        - 至少包含一个大写字母
        - 至少包含一个数字
        """
        if not any(c.isupper() for c in v):
            raise ValueError("密码必须包含至少一个大写字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码必须包含至少一个数字")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str | None) -> str | None:
        """验证全名"""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("全名不能为空")
        return v.strip() if v else None


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(..., min_length=3, max_length=50, description="邮箱")
    password: str = Field(..., min_length=6, description="密码")


# ============== 响应模型 ==============


class TokenResponse(BaseModel):
    """Token 响应"""

    access_token: str = Field(..., description="访问令牌")
    token_type: Literal["bearer"] = Field("bearer", description="令牌类型")
    expires_at: str | None = Field(None, description="过期时间（ISO 8601）")


class UserResponse(BaseModel):
    """用户响应"""

    id: int = Field(..., description="用户 ID")
    email: str = Field(..., description="邮箱")
    full_name: str | None = Field(None, description="全名")
    is_active: bool = Field(..., description="是否激活")
    is_superuser: bool = Field(..., description="是否超级用户")
    created_at: datetime | None = Field(None, description="创建时间")


class UserWithTokenResponse(UserResponse):
    """用户响应（含 Token）"""

    access_token: str = Field(..., description="访问令牌")
    token_type: Literal["bearer"] = Field("bearer", description="令牌类型")


# ============== 会话模型 ==============


class SessionResponse(BaseModel):
    """会话响应"""

    session_id: str = Field(..., description="会话 ID")
    name: str = Field(..., description="会话名称")
    token: str = Field(..., description="会话令牌")
    created_at: str = Field(..., description="创建时间（ISO 8601）")


class SessionListItem(BaseModel):
    """会话列表项"""

    session_id: str = Field(..., description="会话 ID")
    name: str = Field(..., description="会话名称")
    created_at: str = Field(..., description="创建时间（ISO 8601）")
    message_count: int = Field(0, ge=0, description="消息数量")


# ============== API Key 模型 ==============


class APIKeyRequest(BaseModel):
    """API Key 创建请求"""

    name: str = Field(..., min_length=1, max_length=100, description="名称")
    expires_days: int | None = Field(None, ge=1, le=365, description="过期天数")


class APIKeyResponse(BaseModel):
    """API Key 响应"""

    key: str = Field(..., description="API Key")
    name: str = Field(..., description="名称")
    created_at: str = Field(..., description="创建时间（ISO 8601）")
    expires_at: str | None = Field(None, description="过期时间（ISO 8601）")


# ============== 带统一包装的响应类型 ==============


class AuthResultResponse(ApiResponse[UserWithTokenResponse]):
    """认证结果响应（带统一包装）"""

    pass


class TokenResultResponse(ApiResponse[TokenResponse]):
    """Token 结果响应（带统一包装）"""

    pass


# ============== Token 刷新 ==============


class TokenRefreshRequest(BaseModel):
    """Token 刷新请求"""

    refresh_token: str = Field(..., description="刷新令牌")


class TokenRefreshResponse(BaseModel):
    """Token 刷新响应"""

    access_token: str = Field(..., description="新的访问令牌")
    token_type: Literal["bearer"] = Field("bearer", description="令牌类型")
    expires_at: str | None = Field(None, description="过期时间（ISO 8601）")


# ============== Token 验证 ==============


class TokenValidateRequest(BaseModel):
    """Token 验证请求"""

    token: str = Field(..., description="要验证的令牌")


class TokenValidateResponse(BaseModel):
    """Token 验证响应"""

    valid: bool = Field(..., description="是否有效")
    user_id: int | None = Field(None, description="用户 ID（仅在有效时返回）")
    expires_at: str | None = Field(None, description="过期时间（ISO 8601）")
    message: str | None = Field(None, description="状态消息")


# ============== 密码修改 ==============


class ChangePasswordRequest(BaseModel):
    """密码修改请求"""

    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., min_length=8, max_length=100, description="新密码")

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """验证新密码复杂度"""
        if not any(c.isupper() for c in v):
            raise ValueError("新密码必须包含至少一个大写字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("新密码必须包含至少一个数字")
        return v
