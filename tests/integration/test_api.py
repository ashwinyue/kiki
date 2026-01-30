"""API 集成测试"""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """测试健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root(client: TestClient):
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "environment" in data


def test_chat_endpoint(client: TestClient):
    """测试聊天端点"""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "session_id" in data
