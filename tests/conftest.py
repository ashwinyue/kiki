"""测试配置

提供共享的 pytest fixtures 用于单元测试、集成测试和 E2E 测试。
"""

import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

# 设置测试环境变量
os.environ.setdefault("KIKI_ENV", "testing")
os.environ.setdefault("KIKI_LLM_PROVIDER", "mock")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("KIKI_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("KIKI_REDIS_URL", "redis://localhost:16379/15")


@pytest.fixture(scope="session")
def test_settings():
    """测试环境配置 - session 级别，整个测试会话只创建一次"""
    from app.core.config import Settings

    return Settings(_env_file=".env.testing")


@pytest.fixture
def client(test_settings) -> TestClient:
    """测试客户端 - 每个测试独立创建"""
    from fastapi.testclient import TestClient
    from app.main import app

    # 设置测试环境变量
    os.environ["KIKI_ENV"] = "testing"

    return TestClient(app)


@pytest.fixture
def test_user() -> dict:
    """测试用户数据"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }


@pytest.fixture
def test_admin_user() -> dict:
    """测试管理员用户数据"""
    return {
        "username": "admin",
        "email": "admin@example.com",
        "password": "adminpass123",
        "is_admin": True,
    }


@pytest.fixture
def mock_llm_service():
    """Mock LLM 服务 - 避免真实 API 调用"""
    from app.llm import LLMService

    mock_service = MagicMock(spec=LLMService)
    mock_llm = MagicMock()

    # 配置 mock LLM 的基本行为
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="这是一个测试响应")
    )
    mock_llm.invoke = MagicMock(
        return_value=AIMessage(content="这是一个测试响应")
    )
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
    mock_llm.with_retry = MagicMock(return_value=mock_llm)

    mock_service.get_llm = MagicMock(return_value=mock_llm)
    mock_service.get_raw_llm = MagicMock(return_value=mock_llm)
    mock_service._llm = mock_llm
    mock_service._raw_llm = mock_llm

    return mock_service


@pytest.fixture
def mock_llm_with_structured_output():
    """Mock LLM 服务 - 带结构化输出"""
    from pydantic import BaseModel

    class TestDecision(BaseModel):
        """测试决策模型"""
        agent: str = "test_agent"
        reason: str = "测试原因"
        confidence: float = 0.9

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=TestDecision())
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)

    return mock_llm


@pytest.fixture
def sample_messages():
    """示例消息列表"""
    return [
        HumanMessage(content="你好，请问你能帮我什么？"),
        AIMessage(content="我可以帮你解答问题、提供信息和协助完成任务。"),
        HumanMessage(content="那请帮我介绍一下 Python。"),
    ]


@pytest.fixture
def sample_agent_state(sample_messages):
    """示例 Agent 状态"""
    from app.agent.state import AgentState

    return AgentState({
        "messages": sample_messages,
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "iteration_count": 0,
        "max_iterations": 10,
    })


@pytest.fixture
def sample_tools():
    """示例工具列表"""
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return str(result)
        except Exception:
            return "计算错误"

    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        return f"{city}今天天气晴朗，温度 25°C"

    @tool
    def search(query: str) -> str:
        """搜索信息"""
        return f"关于'{query}'的搜索结果..."

    return [calculate, get_weather, search]


@pytest.fixture
def mock_redis():
    """Mock Redis 客户端"""
    mock_client = MagicMock()
    mock_client.get = MagicMock(return_value=None)
    mock_client.set = MagicMock(return_value=True)
    mock_client.delete = MagicMock(return_value=1)
    mock_client.exists = MagicMock(return_value=0)
    mock_client.expire = MagicMock(return_value=True)
    mock_client.keys = MagicMock(return_value=[])
    mock_client.ping = MagicMock(return_value=True)

    # 异步方法
    mock_client.async_get = AsyncMock(return_value=None)
    mock_client.async_set = AsyncMock(return_value=True)
    mock_client.async_delete = AsyncMock(return_value=1)

    return mock_client


@pytest.fixture
def mock_db_session():
    """Mock 数据库会话"""
    from sqlalchemy.orm import sessionmaker

    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.rollback = MagicMock()
    mock_session.close = MagicMock()
    mock_session.query = MagicMock(return_value=MagicMock())
    mock_session.execute = MagicMock(return_value=MagicMock())
    mock_session.refresh = MagicMock()

    return mock_session


@pytest.fixture
async def async_db_session(test_settings):
    """异步数据库会话 - 使用内存 SQLite 进行测试"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.models.database import Base

    # 使用内存 SQLite
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 提供会话
    async with async_session() as session:
        yield session

    # 清理
    await engine.dispose()


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录 - pytest 的 tmp_path 的别名，语义更清晰"""
    return tmp_path


@pytest.fixture
def mock_checkpoint():
    """Mock 检查点"""
    from langgraph.checkpoint import Checkpoint

    return Checkpoint(
        v=1,
        id="test_checkpoint_id",
        channel_values={"messages": []},
        channel_versions={},
        versions_seen={},
    )


@pytest.fixture
def sample_config():
    """示例 LangGraph 配置"""
    return {
        "configurable": {
            "thread_id": "test_thread_123",
            "user_id": "test_user_456",
        }
    }


# ============== 测试标记 ==============

def pytest_configure(config):
    """配置 pytest 自定义标记"""
    config.addinivalue_line("markers", "slow: 标记慢速测试")
    config.addinivalue_line("markers", "integration: 标记集成测试")
    config.addinivalue_line("markers", "unit: 标记单元测试")
    config.addinivalue_line("markers", "e2e: 标记端到端测试")
    config.addinivalue_line("markers", "redis: 标记需要 Redis 的测试")
    config.addinivalue_line("markers", "postgres: 标记需要 PostgreSQL 的测试")
    config.addinivalue_line("markers", "llm: 标记需要真实 LLM API 的测试")
    config.addinivalue_line("markers", "skip_ci: 在 CI 环境中跳过")


# ============== 测试钩子 ==============

@pytest.fixture(autouse=True)
def reset_global_state():
    """每个测试后重置全局状态"""
    yield
    # 这里可以添加需要重置的全局状态
    # 例如：清除全局注册表、重置单例等
