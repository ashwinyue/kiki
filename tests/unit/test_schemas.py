"""Schema 模式测试

测试认证和聊天相关的 Pydantic 模式。
"""

import pytest
from pydantic import ValidationError

from app.schemas.auth import (
    APIKeyRequest,
    APIKeyResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    User,
)
from app.schemas.chat import (
    ChatHistory,
    ChatRequest,
    ChatResponse,
    Message,
)


class TestAuthSchemas:
    """认证 Schema 测试"""

    def test_login_request_valid(self) -> None:
        """测试有效的登录请求"""
        data = LoginRequest(username="testuser", password="password123")
        assert data.username == "testuser"
        assert data.password == "password123"

    def test_login_request_username_too_short(self) -> None:
        """测试用户名太短"""
        with pytest.raises(ValidationError):
            LoginRequest(username="ab", password="password123")

    def test_login_request_password_too_short(self) -> None:
        """测试密码太短"""
        with pytest.raises(ValidationError):
            LoginRequest(username="testuser", password="short")

    def test_register_request_valid(self) -> None:
        """测试有效的注册请求"""
        data = RegisterRequest(
            email="user@example.com",
            username="testuser",
            password="password123",
        )
        assert data.email == "user@example.com"
        assert data.username == "testuser"

    def test_register_request_invalid_email(self) -> None:
        """测试无效邮箱"""
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="not-an-email",
                username="testuser",
                password="password123",
            )

    def test_token_response_valid(self) -> None:
        """测试有效的 Token 响应"""
        data = TokenResponse(
            access_token="test_token",
            refresh_token="refresh_token",
            expires_in=3600,
        )
        assert data.access_token == "test_token"
        assert data.refresh_token == "refresh_token"
        assert data.expires_in == 3600
        assert data.token_type == "bearer"

    def test_user_valid(self) -> None:
        """测试有效的用户信息"""
        data = User(
            id="123",
            username="testuser",
            email="user@example.com",
            created_at="2024-01-01T00:00:00",
        )
        assert data.id == "123"
        assert data.username == "testuser"
        assert data.email == "user@example.com"

    def test_user_email_optional(self) -> None:
        """测试邮箱是可选的"""
        data = User(
            id="123",
            username="testuser",
            created_at="2024-01-01T00:00:00",
        )
        assert data.email is None

    def test_api_key_request_valid(self) -> None:
        """测试有效的 API Key 请求"""
        data = APIKeyRequest(name="Test Key", expires_days=30)
        assert data.name == "Test Key"
        assert data.expires_days == 30

    def test_api_key_request_expires_too_short(self) -> None:
        """测试过期天数太短"""
        with pytest.raises(ValidationError):
            APIKeyRequest(name="Test Key", expires_days=0)

    def test_api_key_request_expires_too_long(self) -> None:
        """测试过期天数太长"""
        with pytest.raises(ValidationError):
            APIKeyRequest(name="Test Key", expires_days=400)

    def test_api_key_response_valid(self) -> None:
        """测试有效的 API Key 响应"""
        data = APIKeyResponse(
            key="sk_test_key",
            name="Test Key",
            created_at="2024-01-01T00:00:00",
            expires_at="2024-02-01T00:00:00",
        )
        assert data.key == "sk_test_key"
        assert data.name == "Test Key"


class TestChatSchemas:
    """聊天 Schema 测试"""

    def test_chat_request_valid(self) -> None:
        """测试有效的聊天请求"""
        data = ChatRequest(
            message="Hello, how are you?",
            session_id="session_123",
        )
        assert data.message == "Hello, how are you?"
        assert data.session_id == "session_123"

    def test_chat_request_with_optional_fields(self) -> None:
        """测试带可选字段的聊天请求"""
        data = ChatRequest(
            message="Test message",
            session_id="session_123",
            stream=False,
        )
        assert data.stream is False

    def test_chat_response_valid(self) -> None:
        """测试有效的聊天响应"""
        data = ChatResponse(
            message="Hello! How can I help you?",
            session_id="session_123",
            finished=True,
        )
        assert data.message == "Hello! How can I help you?"
        assert data.session_id == "session_123"
        assert data.finished is True

    def test_chat_history_valid(self) -> None:
        """测试有效的聊天历史"""
        data = ChatHistory(
            session_id="session_123",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )
        assert data.session_id == "session_123"
        assert len(data.messages) == 2

    def test_message_valid(self) -> None:
        """测试有效的消息"""
        data = Message(role="user", content="Hello, world!")
        assert data.role == "user"
        assert data.content == "Hello, world!"


@pytest.mark.parametrize("username,password,should_pass", [
    ("testuser", "password123", True),
    ("ab", "password123", False),  # username too short (less than 3)
    ("testuser", "short", False),  # password too short (less than 6)
])
def test_login_request_validation(username: str, password: str, should_pass: bool) -> None:
    """参数化测试登录请求验证"""
    if should_pass:
        data = LoginRequest(username=username, password=password)
        assert data.username == username
    else:
        with pytest.raises(ValidationError):
            LoginRequest(username=username, password=password)


@pytest.mark.parametrize("email,should_pass", [
    ("user@example.com", True),
    ("not-an-email", False),
    ("", False),
    ("@example.com", False),
])
def test_email_validation(email: str, should_pass: bool) -> None:
    """参数化测试邮箱验证"""
    if should_pass:
        data = RegisterRequest(
            email=email,
            username="testuser",
            password="password123",
        )
        assert data.email == email
    else:
        with pytest.raises(ValidationError):
            RegisterRequest(
                email=email,
                username="testuser",
                password="password123",
            )
