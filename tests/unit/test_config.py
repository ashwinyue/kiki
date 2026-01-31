"""Config 配置测试

测试 Settings 类的加载和验证功能。
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


class TestSettings:
    """Settings 测试"""

    def test_default_values(self) -> None:
        """测试默认值"""
        # 使用最小配置
        settings = Settings(
            llm_api_key="test_key",
        )

        assert settings.llm_api_key == "test_key"
        assert settings.app_name == "Kiki Agent"
        assert settings.app_version == "0.1.0"

    def test_from_environment(self) -> None:
        """测试从环境变量加载"""
        os.environ["KIKI_LLM_API_KEY"] = "env_key"
        os.environ["KIKI_DEBUG"] = "true"

        settings = Settings()

        assert settings.llm_api_key == "env_key"
        assert settings.debug is True

        # 清理
        del os.environ["KIKI_LLM_API_KEY"]
        del os.environ["KIKI_DEBUG"]

    def test_debug_boolean_conversion(self) -> None:
        """测试 debug 布尔值转换"""
        import os
        # 清除环境变量并跳过 .env 文件以确保构造函数的值生效
        old_debug = os.environ.pop("KIKI_DEBUG", None)
        try:
            # 使用 production 环境，因为 model_post_init 在 development 强制设置 debug=True
            settings_prod = Settings(llm_api_key="test", debug=True, environment="production", secret_key="a" * 32, _env_file=None)
            # model_post_init 会强制 production 的 debug 为 False
            assert settings_prod.debug is False

            settings_test = Settings(llm_api_key="test", debug=True, environment="test", _env_file=None)
            # test 环境不会覆盖 debug
            assert settings_test.debug is True

            settings_test_false = Settings(llm_api_key="test", debug=False, environment="test", _env_file=None)
            assert settings_test_false.debug is False
        finally:
            if old_debug is not None:
                os.environ["KIKI_DEBUG"] = old_debug

    def test_production_secret_key_validation(self) -> None:
        """测试生产环境 secret_key 验证"""
        # secret_key 验证在 production 环境生效
        with pytest.raises(ValueError):
            Settings(llm_api_key="test", environment="production", _env_file=None)

        # 正常的 secret_key 应该通过
        settings = Settings(llm_api_key="test", environment="production", secret_key="a" * 32, _env_file=None)
        assert settings.secret_key == "a" * 32

    def test_database_url_validation(self) -> None:
        """测试数据库 URL 验证"""
        settings = Settings(llm_api_key="test_key", database_url="postgresql://localhost/test")
        assert settings.database_url == "postgresql://localhost/test"

    def test_redis_url_default(self) -> None:
        """测试 Redis URL 默认值"""
        settings = Settings(llm_api_key="test_key")
        assert settings.redis_url is not None

    @pytest.mark.asyncio
    async def test_is_development_property(self) -> None:
        """测试环境判断属性"""
        settings = Settings(llm_api_key="test", environment="development")
        assert settings.is_development is True
        assert settings.is_production is False

        # Production 环境需要有效的 secret_key
        settings_prod = Settings(llm_api_key="test", environment="production", secret_key="a" * 32)
        assert settings_prod.is_production is True
        assert settings_prod.is_development is False

    def test_model_validation(self) -> None:
        """测试模型配置"""
        settings = Settings(llm_api_key="test", llm_model="custom-model")
        assert settings.llm_model == "custom-model"


@pytest.mark.parametrize("env,expected_dev,expected_prod", [
    ("development", True, False),
    ("staging", False, False),
    ("production", False, True),
    ("test", False, False),
])
def test_environment_detection(env: str, expected_dev: bool, expected_prod: bool) -> None:
    """参数化测试环境检测"""
    # 使用 environment 字段名而不是 env
    # production 环境需要有效的 secret_key
    if env == "production":
        settings = Settings(llm_api_key="test", environment=env, secret_key="a" * 32)
    else:
        settings = Settings(llm_api_key="test", environment=env)
    assert settings.is_development == expected_dev
    assert settings.is_production == expected_prod


@pytest.mark.parametrize("debug_value,expected", [
    (True, True),
    (False, False),
])
def test_debug_boolean_values(debug_value: bool, expected: bool) -> None:
    """参数化测试 debug 布尔值"""
    import os
    # 清除环境变量并跳过 .env 文件以确保构造函数的值生效
    # 使用 test 环境而不是 development，因为 development 强制 debug=True
    old_debug = os.environ.pop("KIKI_DEBUG", None)
    try:
        settings = Settings(llm_api_key="test", debug=debug_value, environment="test", _env_file=None)
        assert settings.debug == expected
    finally:
        if old_debug is not None:
            os.environ["KIKI_DEBUG"] = old_debug
