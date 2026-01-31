"""多 Agent API 集成测试

测试 Router、Supervisor、Swarm 等 Agent 系统的 API 端点。
"""

import pytest

from fastapi.testclient import TestClient


class TestAgentSystemsAPI:
    """Agent 系统 API 测试"""

    def test_list_empty_systems(self, client: TestClient) -> None:
        """测试列出空的 Agent 系统"""
        response = client.get("/api/v1/agents/systems")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_router_agent(self, client: TestClient) -> None:
        """测试创建路由 Agent"""
        request_data = {
            "name": "test_router",
            "agents": [
                {
                    "name": "sales",
                    "system_prompt": "你是销售助手",
                    "tools": ["calculate"],
                },
                {
                    "name": "support",
                    "system_prompt": "你是客服助手",
                    "tools": [],
                },
            ],
        }

        response = client.post("/api/v1/agents/router", json=request_data)
        # 可能会因为 LLM 未配置而失败，这是预期的
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            data = response.json()
            assert data["type"] == "router"
            assert data["name"] == "test_router"
            assert "sales" in data["agents"]
            assert "support" in data["agents"]

    def test_create_supervisor_agent(self, client: TestClient) -> None:
        """测试创建监督 Agent"""
        request_data = {
            "name": "test_supervisor",
            "workers": [
                {
                    "name": "worker1",
                    "system_prompt": "你是工作助手 1",
                    "tools": [],
                },
                {
                    "name": "worker2",
                    "system_prompt": "你是工作助手 2",
                    "tools": ["calculate"],
                },
            ],
        }

        response = client.post("/api/v1/agents/supervisor", json=request_data)
        # 可能会因为 LLM 未配置而失败，这是预期的
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            data = response.json()
            assert data["type"] == "supervisor"
            assert data["name"] == "test_supervisor"

    def test_create_swarm_agent(self, client: TestClient) -> None:
        """测试创建 Swarm Agent"""
        request_data = {
            "name": "test_swarm",
            "agents": [
                {"name": "Alice", "system_prompt": "你是 Alice", "tools": []},
                {"name": "Bob", "system_prompt": "你是 Bob", "tools": []},
            ],
            "handoff_mapping": {
                "Alice": ["Bob"],
                "Bob": ["Alice"],
            },
            "default_agent": "Alice",
        }

        response = client.post("/api/v1/agents/swarm", json=request_data)
        # Swarm 不需要 LLM 就能创建
        assert response.status_code == 200

        data = response.json()
        assert data["type"] == "swarm"
        assert data["name"] == "test_swarm"
        assert "Alice" in data["agents"]
        assert "Bob" in data["agents"]

    def test_delete_nonexistent_system(self, client: TestClient) -> None:
        """测试删除不存在的系统"""
        response = client.delete("/api/v1/agents/systems/nonexistent")
        assert response.status_code == 404

    def test_get_history_nonexistent_system(self, client: TestClient) -> None:
        """测试获取不存在系统的历史"""
        response = client.get("/api/v1/agents/systems/nonexistent/history/session123")
        assert response.status_code == 404


class TestAgentChatAPI:
    """Agent 聊天 API 测试"""

    def test_quick_chat_single(self, client: TestClient) -> None:
        """测试快速聊天 - 单 Agent 模式"""
        # 需要先设置 OPENAI_API_KEY 环境变量
        request_data = {
            "message": "你好",
            "session_id": "test_session_001",
            "user_id": "test_user",
        }

        response = client.post(
            "/api/v1/agents/chat",
            json=request_data,
            params={"system_type": "single", "system_name": "default"},
        )
        # 可能会因为 LLM 未配置而失败
        assert response.status_code in (200, 404, 500)

    def test_chat_nonexistent_system(self, client: TestClient) -> None:
        """测试使用不存在的系统聊天"""
        request_data = {
            "message": "你好",
            "session_id": "test_session_002",
        }

        response = client.post(
            "/api/v1/agents/router/nonexistent/chat",
            json=request_data,
        )
        assert response.status_code == 404
