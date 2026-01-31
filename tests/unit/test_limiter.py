"""限流器测试

测试 slowapi 限流器的功能。
"""

from unittest.mock import MagicMock

import pytest
from slowapi.errors import RateLimitExceeded

from app.core.limiter import (
    RateLimit,
    _get_identifier,
    limit,
    limiter,
    rate_limit_exceeded_handler,
)


class TestGetIdentifier:
    """_get_identifier 函数测试"""

    def test_identifier_from_user_id_in_state(self) -> None:
        """测试从请求状态获取用户 ID"""
        request = MagicMock()
        request.state.user_id = "user_123"

        result = _get_identifier(request)

        assert result == "user:user_123"

    def test_identifier_from_x_user_id_header(self) -> None:
        """测试从请求头获取用户 ID"""
        request = MagicMock()
        request.state = MagicMock()
        request.state.user_id = None
        request.headers.get = MagicMock(return_value="user_456")

        result = _get_identifier(request)

        assert result == "user:user_456"

    def test_identifier_from_ip_address(self) -> None:
        """测试使用 IP 地址"""
        request = MagicMock()
        request.state = MagicMock()
        request.state.user_id = None
        request.headers.get = MagicMock(return_value=None)

        with patch("app.core.limiter.get_remote_address") as mock_get_remote:
            mock_get_remote.return_value = "127.0.0.1"

            result = _get_identifier(request)

            assert result == "127.0.0.1"


class TestRateLimit:
    """RateLimit 类测试"""

    def test_chat_limits(self) -> None:
        """测试聊天限流规则"""
        assert RateLimit.CHAT == ["30 per minute", "500 per day"]

    def test_chat_stream_limits(self) -> None:
        """测试流式聊天限流规则"""
        assert RateLimit.CHAT_STREAM == ["20 per minute", "300 per day"]

    def test_register_limits(self) -> None:
        """测试注册限流规则"""
        assert RateLimit.REGISTER == ["10 per hour", "50 per day"]

    def test_login_limits(self) -> None:
        """测试登录限流规则"""
        assert RateLimit.LOGIN == ["20 per minute", "100 per day"]

    def test_api_limits(self) -> None:
        """测试 API 限流规则"""
        assert RateLimit.API == ["100 per minute", "1000 per day"]

    def test_health_limits(self) -> None:
        """测试健康检查限流规则"""
        assert RateLimit.HEALTH == ["20 per minute"]


class TestLimitDecorator:
    """limit 装饰器测试"""

    def test_limit_decorator(self) -> None:
        """测试限流装饰器"""
        decorator = limit("10 per minute")

        assert decorator is not None

    def test_limit_with_per_method(self) -> None:
        """测试按方法分别限流"""
        decorator = limit("10 per minute", per_method=True)

        assert decorator is not None

    def test_limit_with_per_method_false(self) -> None:
        """测试不按方法分别限流"""
        decorator = limit("10 per minute", per_method=False)

        assert decorator is not None


class TestRateLimitExceededHandler:
    """rate_limit_exceeded_handler 测试"""

    @pytest.mark.asyncio
    async def test_handler_response(self) -> None:
        """测试处理器返回响应"""
        request = MagicMock()
        request.url.path = "/api/test"

        exc = RateLimitExceeded("10 per minute")
        exc.description = "10 per minute"

        response = await rate_limit_exceeded_handler(request, exc)

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.body.decode()


class TestLimiter:
    """limiter 实例测试"""

    def test_limiter_attributes(self) -> None:
        """测试限流器属性"""
        assert hasattr(limiter, "key_func")
        assert hasattr(limiter, "default_limits")

    def test_limiter_check(self) -> None:
        """测试限流器检查"""
        # limiter 应该有 check 方法
        assert hasattr(limiter, "check")


@pytest.mark.parametrize("limit_value", [
    "10 per minute",
    "100 per hour",
    "1000 per day",
    "5 per second",
])
def test_various_limit_values(limit_value: str) -> None:
    """参数化测试各种限流值"""
    decorator = limit(limit_value)
    assert decorator is not None


@pytest.mark.parametrize("limit_rule", [
    RateLimit.CHAT,
    RateLimit.CHAT_STREAM,
    RateLimit.REGISTER,
    RateLimit.LOGIN,
    RateLimit.API,
    RateLimit.HEALTH,
])
def test_predefined_rate_limits(limit_rule) -> None:
    """参数化测试预定义限流规则"""
    assert isinstance(limit_rule, list)
    assert len(limit_rule) > 0
