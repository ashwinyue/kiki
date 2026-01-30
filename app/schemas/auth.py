"""认证相关模式"""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class RegisterRequest(BaseModel):
    """注册请求"""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """令牌响应"""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


class User(BaseModel):
    """用户信息"""

    id: str
    username: str
    email: str | None = None
    created_at: str


class APIKeyRequest(BaseModel):
    """API Key 创建请求"""

    name: str = Field(..., min_length=1, max_length=100)
    expires_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API Key 响应"""

    key: str
    name: str
    created_at: str
    expires_at: str | None = None
