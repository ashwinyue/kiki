"""LLM 服务测试

测试 LLMService 和 LLMRegistry 的核心功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.llm import (
    LLMRegistry,
    LLMService,
    get_llm_service,
)


class MockRateLimitError(Exception):
    """Mock RateLimitError for testing"""
    pass


class TestLLMRegistry:
    """LLMRegistry 测试"""

    def setup_method(self) -> None:
        """每个测试前清空注册表"""
        LLMRegistry._models.clear()
        LLMRegistry._initialized = False

    def teardown_method(self) -> None:
        """每个测试后清空注册表"""
        LLMRegistry._models.clear()
        LLMRegistry._initialized = False

    def test_register_llm(self) -> None:
        """测试注册 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        LLMRegistry.register("test-model", llm, "Test Model")

        assert "test-model" in LLMRegistry._models
        assert LLMRegistry._models["test-model"]["llm"] is llm
        assert LLMRegistry._models["test-model"]["description"] == "Test Model"

    def test_get_registered_llm(self) -> None:
        """测试获取已注册的 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        LLMRegistry.register("test-model", llm)

        result = LLMRegistry.get("test-model")
        assert result is llm

    def test_get_nonexistent_llm(self) -> None:
        """测试获取不存在的 LLM"""
        with pytest.raises(ValueError, match="未注册"):
            LLMRegistry.get("nonexistent-model")

    def test_get_with_kwargs(self) -> None:
        """测试使用自定义参数获取 LLM"""
        # 使用 BaseChatModel 而不是 spec=ChatOpenAI，避免 isinstance 检查
        llm = MagicMock()
        llm.model_name = "gpt-4o"
        LLMRegistry.register("gpt-4o", llm)

        result = LLMRegistry.get("gpt-4o", temperature=0.5)
        # 对于非 ChatOpenAI 类型，应该返回原始 llm
        assert result is llm

    def test_list_models(self) -> None:
        """测试列出所有模型"""
        llm1 = MagicMock(spec=ChatOpenAI)
        llm2 = MagicMock(spec=ChatOpenAI)

        LLMRegistry.register("model1", llm1)
        LLMRegistry.register("model2", llm2)

        models = LLMRegistry.list_models()
        assert set(models) == {"model1", "model2"}

    def test_is_registered(self) -> None:
        """测试检查模型是否已注册"""
        llm = MagicMock(spec=ChatOpenAI)
        LLMRegistry.register("test-model", llm)

        assert LLMRegistry.is_registered("test-model") is True
        assert LLMRegistry.is_registered("nonexistent") is False


class TestLLMService:
    """LLMService 测试"""

    def setup_method(self) -> None:
        """每个测试前重置服务"""
        # 重置全局服务
        import app.llm
        app.llm._llm_service = None

    def teardown_method(self) -> None:
        """每个测试后清理"""
        import app.llm
        app.llm._llm_service = None

    def test_init_with_default_model(self) -> None:
        """测试使用默认模型初始化"""
        llm = MagicMock(spec=ChatOpenAI)
        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model", max_retries=2)

            assert service.current_model == "test-model"
            assert service._max_retries == 2

    def test_get_raw_llm(self) -> None:
        """测试获取原始 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            raw_llm = service.get_raw_llm()
            assert raw_llm is not None

    def test_get_llm_with_retry(self) -> None:
        """测试获取带重试的 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            llm_with_retry = service.get_llm_with_retry()
            assert llm_with_retry is not None

    def test_bind_tools(self) -> None:
        """测试绑定工具"""
        llm = MagicMock(spec=ChatOpenAI)
        llm.bind_tools = MagicMock(return_value=llm)

        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            tools = [MagicMock()]
            result = service.bind_tools(tools)

            assert result is service  # 链式调用

    @pytest.mark.asyncio
    async def test_call_success(self) -> None:
        """测试成功调用 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        llm_with_retry = MagicMock()
        llm_with_retry.ainvoke = AsyncMock(
            return_value=AIMessage(content="Test response")
        )

        with patch.object(LLMRegistry, "get", return_value=llm), \
             patch.object(LLMRegistry, "list_models", return_value=["test-model"]):

            llm.with_retry = MagicMock(return_value=llm_with_retry)

            service = LLMService(default_model="test-model")
            service._llm = llm_with_retry

            messages = [HumanMessage(content="Hello")]
            response = await service.call(messages)

            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_call_with_model_switch(self) -> None:
        """测试模型切换"""
        llm1 = MagicMock(spec=ChatOpenAI)
        llm2 = MagicMock(spec=ChatOpenAI)

        llm_with_retry1 = MagicMock()
        llm_with_retry1.ainvoke = AsyncMock(
            side_effect=MockRateLimitError("Rate limited")
        )

        llm_with_retry2 = MagicMock()
        llm_with_retry2.ainvoke = AsyncMock(
            return_value=AIMessage(content="Response from model 2")
        )

        with patch.object(LLMRegistry, "get", side_effect=[llm1, llm2]), \
             patch.object(LLMRegistry, "list_models", return_value=["model1", "model2"]):

            llm1.with_retry = MagicMock(return_value=llm_with_retry1)
            llm2.with_retry = MagicMock(return_value=llm_with_retry2)

            service = LLMService(default_model="model1")
            service._llm = llm_with_retry1

            messages = [HumanMessage(content="Hello")]
            response = await service.call(messages)

            assert response.content == "Response from model 2"

    def test_with_structured_output(self) -> None:
        """测试结构化输出"""
        llm = MagicMock(spec=ChatOpenAI)

        class TestSchema(BaseModel):
            answer: str

        structured_llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured_llm)

        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            result = service.with_structured_output(TestSchema)
            assert result is not None

    def test_apply_retry(self) -> None:
        """测试应用重试配置"""
        llm = MagicMock(spec=ChatOpenAI)
        llm_with_retry = MagicMock()

        llm.with_retry = MagicMock(return_value=llm_with_retry)

        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            result = service._apply_retry(llm)

            # with_retry 被调用了：一次在构造时，一次在 _apply_retry
            assert llm.with_retry.call_count >= 1
            assert result is llm_with_retry

    def test_get_llm_with_tools(self) -> None:
        """测试获取带工具的 LLM"""
        llm = MagicMock(spec=ChatOpenAI)
        bound_llm = MagicMock()

        llm.bind_tools = MagicMock(return_value=bound_llm)

        with patch.object(LLMRegistry, "get", return_value=llm):
            service = LLMService(default_model="test-model")

            tools = [MagicMock()]
            result = service.get_llm_with_tools(tools)

            assert result is not None


class TestGetLLMService:
    """get_llm_service 单例测试"""

    def teardown_method(self) -> None:
        """每个测试后重置单例"""
        from app.llm import service
        service._llm_service = None

    def test_singleton(self) -> None:
        """测试单例模式"""
        with patch("app.llm.service.LLMService") as mock_service_class:
            mock_instance = MagicMock()
            mock_service_class.return_value = mock_instance

            service1 = get_llm_service()
            service2 = get_llm_service()

            assert service1 is service2
            mock_service_class.assert_called_once()


@pytest.mark.parametrize("max_retries", [1, 2, 5])
def test_llm_service_different_retries(max_retries: int) -> None:
    """参数化测试不同重试次数"""
    llm = MagicMock(spec=ChatOpenAI)

    with patch.object(LLMRegistry, "get", return_value=llm):
        service = LLMService(max_retries=max_retries)
        assert service._max_retries == max_retries


@pytest.mark.asyncio
async def test_llm_service_call_all_models_failed() -> None:
    """测试所有模型都失败的情况"""
    llm = MagicMock(spec=ChatOpenAI)
    llm_with_retry = MagicMock()
    llm_with_retry.ainvoke = AsyncMock(
        side_effect=MockRateLimitError("All rate limited")
    )

    with patch.object(LLMRegistry, "get", return_value=llm), \
         patch.object(LLMRegistry, "list_models", return_value=["model1"]):

        llm.with_retry = MagicMock(return_value=llm_with_retry)

        service = LLMService(default_model="model1")
        service._llm = llm_with_retry

        messages = [HumanMessage(content="Hello")]

        with pytest.raises(RuntimeError, match="LLM 调用失败"):
            await service.call(messages)
