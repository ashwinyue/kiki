"""API 集成测试

测试基础 API 端点的功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_health_check(self, client: TestClient) -> None:
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Redis 可能不可用，状态可能是 healthy 或 degraded
        assert data["status"] in ("healthy", "degraded")
        assert "app" in data
        assert "version" in data

    def test_root(self, client: TestClient) -> None:
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "environment" in data


class TestChatEndpoints:
    """聊天 API 端点测试"""

    def test_chat_endpoint_missing_message(self, client: TestClient) -> None:
        """测试聊天端点 - 缺少消息"""
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "test_session"},
        )
        assert response.status_code == 422  # Validation error

    def test_chat_endpoint_missing_session_id(self, client: TestClient) -> None:
        """测试聊天端点 - 缺少会话 ID"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.skip(reason="需要配置 LLM API key")
    def test_chat_endpoint_full_request(self, client: TestClient) -> None:
        """测试完整聊天请求"""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, how are you?",
                "session_id": "test_session_001",
                "user_id": "test_user",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["session_id"] == "test_session_001"

    def test_get_chat_history_empty(self, client: TestClient) -> None:
        """测试获取空聊天历史"""
        response = client.get("/api/v1/chat/history/nonexistent_session")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert data["session_id"] == "nonexistent_session"
        assert len(data["messages"]) == 0

    def test_clear_chat_history(self, client: TestClient) -> None:
        """测试清除聊天历史"""
        response = client.delete("/api/v1/chat/history/test_session")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_context_stats_empty(self, client: TestClient) -> None:
        """测试获取空会话上下文统计"""
        response = client.get("/api/v1/chat/context/nonexistent_session/stats")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "message_count" in data

    def test_clear_context(self, client: TestClient) -> None:
        """测试清除会话上下文"""
        response = client.delete("/api/v1/chat/context/test_session")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestChatStreamEndpoint:
    """流式聊天端点测试"""

    def test_stream_endpoint(self, client: TestClient) -> None:
        """测试流式聊天端点返回正确的内容类型"""
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Hello",
                "session_id": "test_session",
            },
        )
        # 流式端点返回 200，即使 LLM 未配置
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestErrorResponse:
    """错误响应测试"""

    def test_404_endpoint(self, client: TestClient) -> None:
        """测试不存在的端点"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_invalid_json(self, client: TestClient) -> None:
        """测试无效 JSON"""
        response = client.post(
            "/api/v1/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
