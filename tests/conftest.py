"""测试配置"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """测试客户端"""
    return TestClient(app)


@pytest.fixture
def settings():
    """测试配置"""
    from app.core.config import Settings
    return Settings()


@pytest.fixture
def test_user():
    """测试用户"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }
