"""E2E 测试配置

提供全链路测试所需的共享 fixtures。
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import Settings, get_settings
from app.core.database import init_db
from app.main import app
from app.models.database import (
    ChatSession,
    Memory,
    Message,
    Thread,
    User,
)
from app.services.database import (
    message_repository,
    session_repository,
    user_repository,
)
from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 测试数据库配置 ==============

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres123!@#@localhost:15432/kiki_test"


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环 - session 级别"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def e2e_engine():
    """E2E 测试数据库引擎"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # 初始化数据库表
    async with engine.begin() as conn:
        await conn.run_sync(lambda conn: None)  # 确保连接

    yield engine

    # 清理
    await engine.dispose()


@pytest.fixture(scope="session")
async def init_e2e_db(e2e_engine):
    """初始化 E2E 测试数据库"""
    async with e2e_engine.begin() as conn:
        # 创建所有表
        await conn.run_sync(lambda conn: None)

    yield

    # 测试结束后清理表数据（保留表结构）
    async with e2e_engine.begin() as conn:
        await conn.exec_driver_sql("DELETE FROM messages")
        await conn.exec_driver_sql("DELETE FROM chatsessions")
        await conn.exec_driver_sql("DELETE FROM threads")
        await conn.exec_driver_sql("DELETE FROM memories")
        await conn.exec_driver_sql("DELETE FROM users")


@pytest.fixture
async def db_session(e2e_engine, init_e2e_db) -> AsyncGenerator[AsyncSession, None]:
    """创建数据库会话 - 每个 test 独立"""
    async_session = async_sessionmaker(
        e2e_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        # 每个测试开始前清理数据
        await session.exec_driver_sql("DELETE FROM messages")
        await session.exec_driver_sql("DELETE FROM chatsessions")
        await session.exec_driver_sql("DELETE FROM threads")
        await session.exec_driver_sql("DELETE FROM memories")
        await session.exec_driver_sql("DELETE FROM users")
        await session.commit()

        yield session

        # 测试结束后回滚
        await session.rollback()


# ============== 测试客户端 ==============


@pytest.fixture
def e2e_client() -> Generator[TestClient, None, None]:
    """E2E 测试客户端"""
    from fastapi.testclient import TestClient

    # 使用测试环境配置
    import os

    os.environ["KIKI_ENV"] = "testing"
    os.environ["KIKI_DATABASE_URL"] = TEST_DATABASE_URL

    with TestClient(app) as client:
        yield client


# ============== 测试用户 Fixtures ==============


@pytest.fixture
async def test_user(db_session: AsyncSession) -> dict[str, Any]:
    """创建测试用户"""
    from app.models.database import UserCreate
    from app.repositories.base import Repository

    repo = user_repository(db_session)

    user_create = UserCreate(
        email="test@example.com",
        password="password123",
        full_name="Test User",
    )

    user = await repo.create_with_password(user_create)
    await db_session.commit()
    await db_session.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "password": "password123",
        "full_name": user.full_name,
    }


@pytest.fixture
async def test_admin_user(db_session: AsyncSession) -> dict[str, Any]:
    """创建测试管理员用户"""
    from app.models.database import UserCreate

    repo = user_repository(db_session)

    user_create = UserCreate(
        email="admin@example.com",
        password="admin123",
        full_name="Admin User",
    )

    user = await repo.create_with_password(user_create)
    user.is_superuser = True
    await db_session.commit()
    await db_session.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "password": "admin123",
        "full_name": user.full_name,
        "is_superuser": True,
    }


@pytest.fixture
def auth_headers(test_user: dict[str, Any]) -> dict[str, str]:
    """生成认证头"""
    from app.auth.jwt import create_access_token

    token_data = create_access_token(data={"sub": str(test_user["id"])})
    return {"Authorization": f"Bearer {token_data.access_token}"}


@pytest.fixture
def auth_headers_sync(e2e_client: TestClient, test_user: dict[str, Any]) -> dict[str, str]:
    """通过登录获取认证头"""
    # 先注册用户
    e2e_client.post(
        "/api/v1/auth/register",
        json={
            "email": test_user["email"],
            "password": test_user["password"],
            "full_name": test_user["full_name"],
        },
    )

    # 登录获取 token
    response = e2e_client.post(
        "/api/v1/auth/login/json",
        json={
            "username": test_user["email"],
            "password": test_user["password"],
        },
    )

    assert response.status_code == 200
    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


# ============== 测试会话 Fixtures ==============


@pytest.fixture
async def test_session(db_session: AsyncSession, test_user: dict[str, Any]) -> ChatSession:
    """创建测试会话"""
    repo = session_repository(db_session)

    from app.models.database import SessionCreate

    session_obj = await repo.create_with_user(
        data=SessionCreate(name="Test Session"),
        user_id=test_user["id"],
    )
    await db_session.commit()
    await db_session.refresh(session_obj)

    return session_obj


# ============== LLM Mock Fixtures ==============


@pytest.fixture
def mock_llm_response():
    """Mock LLM 响应"""
    return {
        "content": "这是一个测试响应。我可以帮助您解答问题、提供信息和协助完成任务。",
        "role": "assistant",
    }


@pytest.fixture
def mock_stream_chunks():
    """Mock 流式响应块"""
    return [
        "这", "是", "一", "个", "测", "试", "响", "应", "。",
        "我", "可", "以", "帮", "助", "您",
    ]


# ============== 测试辅助函数 ==============


@pytest.fixture
def create_user_with_token(e2e_client: TestClient):
    """创建用户并返回 token 的辅助函数"""

    async def _create(email: str, password: str = "password123", full_name: str = "Test User") -> dict[str, Any]:
        response = e2e_client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "full_name": full_name},
        )
        assert response.status_code == 200
        data = response.json()
        return {
            "user": data,
            "token": data["access_token"],
            "headers": {"Authorization": f'Bearer {data["access_token"]}'},
        }

    return _create


@pytest.fixture
def login_user(e2e_client: TestClient):
    """登录用户并返回 token 的辅助函数"""

    def _login(email: str, password: str) -> dict[str, Any]:
        response = e2e_client.post(
            "/api/v1/auth/login/json",
            json={"username": email, "password": password},
        )
        assert response.status_code == 200
        data = response.json()
        return {
            "token": data["access_token"],
            "headers": {"Authorization": f'Bearer {data["access_token"]}'},
        }

    return _login
