"""
Tenacity 重试装饰器单元测试

测试 app/utils/retry_decorators.py 模块的功能。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from app.utils.retry_decorators import (
    NonRetryableError,
    RetryableError,
    create_llm_retry_decorator,
    create_retry_decorator,
    is_retryable_error,
    retry_async,
    retry_sync,
    retry_with_backoff,
)


class TestRetryDecorator:
    """测试基础重试装饰器"""

    @pytest.mark.asyncio
    async def test_async_function_success_on_first_try(self):
        """测试异步函数第一次尝试就成功"""

        @create_retry_decorator(max_attempts=3)
        async def always_succeed() -> str:
            return "success"

        result = await always_succeed()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_function_retry_until_success(self):
        """测试异步函数重试后成功"""
        attempts = [0]

        @create_retry_decorator(max_attempts=3)
        async def fail_twice_then_succeed() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("临时错误")
            return "success after retries"

        result = await fail_twice_then_succeed()
        assert result == "success after retries"
        assert attempts[0] == 3

    @pytest.mark.asyncio
    async def test_async_function_max_attempts_exceeded(self):
        """测试异步函数超过最大重试次数"""

        @create_retry_decorator(max_attempts=2)
        async def always_fail() -> str:
            raise ConnectionError("持续错误")

        with pytest.raises(ConnectionError, match="持续错误"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_async_function_specific_exception_only(self):
        """测试只对特定异常类型重试"""
        attempts = [0]

        @create_retry_decorator(
            max_attempts=3,
            exceptions=(ValueError,),  # 只重试 ValueError
        )
        async def fail_with_different_errors() -> str:
            attempts[0] += 1
            if attempts[0] == 1:
                raise ValueError("可重试错误")
            if attempts[0] == 2:
                raise TypeError("不可重试错误")
            return "success"

        # TypeError 不会重试，直接抛出
        with pytest.raises(TypeError, match="不可重试错误"):
            await fail_with_different_errors()

    def test_sync_function_success_on_first_try(self):
        """测试同步函数第一次尝试就成功"""

        @create_retry_decorator(max_attempts=3)
        def always_succeed() -> str:
            return "success"

        result = always_succeed()
        assert result == "success"

    def test_sync_function_retry_until_success(self):
        """测试同步函数重试后成功"""
        attempts = [0]

        @create_retry_decorator(max_attempts=3)
        def fail_twice_then_succeed() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("临时错误")
            return "success after retries"

        result = fail_twice_then_succeed()
        assert result == "success after retries"
        assert attempts[0] == 3


class TestLLMRetryDecorator:
    """测试 LLM 专用重试装饰器"""

    @pytest.mark.asyncio
    async def test_llm_retry_with_default_settings(self):
        """测试使用默认设置的 LLM 重试"""

        @create_llm_retry_decorator()
        async def mock_llm_call() -> dict:
            return {"result": "response"}

        result = await mock_llm_call()
        assert result == {"result": "response"}

    @pytest.mark.asyncio
    async def test_llm_retry_with_custom_settings(self):
        """测试使用自定义设置的 LLM 重试"""
        attempts = [0]

        @create_llm_retry_decorator(max_attempts=2, min_wait=0.1)
        async def mock_llm_call() -> dict:
            attempts[0] += 1
            if attempts[0] == 1:
                raise ConnectionError("网络错误")
            return {"result": "response"}

        result = await mock_llm_call()
        assert result == {"result": "response"}
        assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_llm_retry_on_timeout(self):
        """测试超时异常重试"""
        attempts = [0]

        @create_llm_retry_decorator(max_attempts=3)
        async def timeout_then_succeed() -> dict:
            attempts[0] += 1
            if attempts[0] < 2:
                raise TimeoutError("请求超时")
            return {"result": "response"}

        result = await timeout_then_succeed()
        assert result == {"result": "response"}


class TestRetryWithBackoff:
    """测试带退避的重试装饰器（简化版）"""

    @pytest.mark.asyncio
    async def test_retry_with_backoff_no_params(self):
        """测试无参数使用"""

        @retry_with_backoff
        async def succeed_immediately() -> str:
            return "success"

        result = await succeed_immediately()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_with_backoff_with_params(self):
        """测试带参数使用"""
        attempts = [0]

        @retry_with_backoff(max_attempts=3, min_wait=0.1)
        async def fail_then_succeed() -> str:
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("错误")
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert attempts[0] == 2


class TestRetryableError:
    """测试 RetryableError 异常"""

    def test_retryable_error_creation(self):
        """测试创建 RetryableError"""
        error = RetryableError("临时错误", retry_after=5.0)
        assert str(error) == "临时错误"
        assert error.retry_after == 5.0

    def test_retryable_error_without_retry_after(self):
        """测试不带 retry_after 的 RetryableError"""
        error = RetryableError("临时错误")
        assert error.retry_after is None

    def test_is_retryable_error_with_retryable(self):
        """测试判断可重试异常"""
        error = RetryableError("临时错误")
        assert is_retryable_error(error) is True

    def test_is_retryable_error_with_non_retryable(self):
        """测试判断不可重试异常"""
        error = NonRetryableError("永久错误")
        assert is_retryable_error(error) is False

    def test_is_retryable_error_with_generic_exception(self):
        """测试判断普通异常"""
        error = ValueError("普通错误")
        assert is_retryable_error(error) is False


class TestRetryAsyncWrapper:
    """测试异步重试包装器"""

    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """测试异步重试包装器成功"""
        async def always_succeed() -> str:
            return "success"

        result = await retry_async(always_succeed)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self):
        """测试异步重试包装器带重试"""
        attempts = [0]

        async def fail_twice() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("错误")
            return "success"

        result = await retry_async(fail_twice, max_attempts=3, min_wait=0.1)
        assert result == "success"
        assert attempts[0] == 3

    @pytest.mark.asyncio
    async def test_retry_async_with_args_and_kwargs(self):
        """测试异步重试包装器传递参数"""
        async def add(a: int, b: int) -> int:
            return a + b

        result = await retry_async(add, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_retry_async_max_attempts_exceeded(self):
        """测试异步重试包装器超过最大尝试次数"""

        async def always_fail() -> str:
            raise ConnectionError("持续错误")

        with pytest.raises(ConnectionError, match="持续错误"):
            await retry_async(always_fail, max_attempts=2, min_wait=0.1)

    @pytest.mark.asyncio
    async def test_retry_async_with_retryable_error_custom_wait(self):
        """测试异步重试包装器使用 RetryableError 的自定义等待时间"""

        async def fail_with_custom_wait() -> str:
            raise RetryableError("自定义等待", retry_after=0.5)

        start = asyncio.get_event_loop().time()
        with pytest.raises(RetryableError):
            await retry_async(fail_with_custom_wait, max_attempts=2, min_wait=2.0)
        elapsed = asyncio.get_event_loop().time() - start

        # 应该使用自定义的 0.5 秒等待，而不是默认的 2.0 秒
        assert 0.4 < elapsed < 1.0


class TestRetrySyncWrapper:
    """测试同步重试包装器"""

    def test_retry_sync_success(self):
        """测试同步重试包装器成功"""
        def always_succeed() -> str:
            return "success"

        result = retry_sync(always_succeed)
        assert result == "success"

    def test_retry_sync_with_retries(self):
        """测试同步重试包装器带重试"""
        attempts = [0]

        def fail_twice() -> str:
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("错误")
            return "success"

        result = retry_sync(fail_twice, max_attempts=3, min_wait=0.1)
        assert result == "success"
        assert attempts[0] == 3

    def test_retry_sync_max_attempts_exceeded(self):
        """测试同步重试包装器超过最大尝试次数"""

        def always_fail() -> str:
            raise ConnectionError("持续错误")

        with pytest.raises(ConnectionError, match="持续错误"):
            retry_sync(always_fail, max_attempts=2, min_wait=0.1)
