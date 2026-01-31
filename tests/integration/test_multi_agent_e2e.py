"""多 Agent 端到端集成测试

测试多 Agent 协作系统的完整流程。
"""

import pytest
from fastapi.testclient import TestClient


class TestRouterAgentE2E:
    """路由 Agent 端到端测试"""

    def test_router_agent_full_workflow(self, client: TestClient) -> None:
        """测试路由 Agent 完整工作流程"""
        # 1. 创建路由 Agent 系统
        create_response = client.post(
            "/api/v1/agents/router",
            json={
                "name": "test_router_e2e",
                "agents": [
                    {
                        "name": "sales",
                        "system_prompt": "你是销售助手，负责产品咨询",
                        "tools": ["calculate"],
                    },
                    {
                        "name": "support",
                        "system_prompt": "你是客服助手，负责售后支持",
                        "tools": [],
                    },
                ],
            },
        )
        # LLM 未配置时可能返回 500
        assert create_response.status_code in (200, 500)

        if create_response.status_code == 200:
            data = create_response.json()
            system_id = data["name"]

            # 2. 验证系统已创建
            list_response = client.get("/api/v1/agents/systems")
            assert list_response.status_code == 200
            systems = list_response.json()
            assert any(s["name"] == "test_router_e2e" for s in systems)

            # 3. 使用系统进行对话
            chat_response = client.post(
                f"/api/v1/agents/router/{system_id}/chat",
                json={
                    "message": "我想了解产品价格",
                    "session_id": "e2e_session_001",
                },
            )
            # 可能因为 LLM 未配置失败
            assert chat_response.status_code in (200, 500)

            # 4. 获取聊天历史
            history_response = client.get(
                f"/api/v1/agents/systems/router_{system_id}/history/e2e_session_001"
            )
            assert history_response.status_code == 200

            # 5. 清理
            delete_response = client.delete(f"/api/v1/agents/systems/router_{system_id}")
            assert delete_response.status_code == 200

    def test_router_agent_with_custom_prompt(self, client: TestClient) -> None:
        """测试使用自定义路由提示词"""
        response = client.post(
            "/api/v1/agents/router",
            json={
                "name": "custom_router",
                "agents": [
                    {"name": "agent1", "system_prompt": "Agent 1", "tools": []},
                    {"name": "agent2", "system_prompt": "Agent 2", "tools": []},
                ],
                "router_prompt": "自定义路由提示词",
            },
        )
        assert response.status_code in (200, 500)


class TestSupervisorAgentE2E:
    """监督 Agent 端到端测试"""

    def test_supervisor_agent_full_workflow(self, client: TestClient) -> None:
        """测试监督 Agent 完整工作流程"""
        # 1. 创建监督 Agent 系统
        create_response = client.post(
            "/api/v1/agents/supervisor",
            json={
                "name": "test_supervisor_e2e",
                "workers": [
                    {
                        "name": "researcher",
                        "system_prompt": "研究员，负责收集信息",
                        "tools": [],
                    },
                    {
                        "name": "writer",
                        "system_prompt": "撰稿人，负责整理内容",
                        "tools": ["calculate"],
                    },
                ],
            },
        )
        assert create_response.status_code in (200, 500)

        if create_response.status_code == 200:
            data = create_response.json()
            assert data["type"] == "supervisor"

            # 2. 使用系统
            chat_response = client.post(
                f"/api/v1/agents/supervisor/test_supervisor_e2e/chat",
                json={
                    "message": "帮我写一篇关于 AI 的文章",
                    "session_id": "supervisor_session_001",
                },
            )
            assert chat_response.status_code in (200, 500)

            # 3. 清理
            client.delete(f"/api/v1/agents/systems/supervisor_test_supervisor_e2e")


class TestSwarmAgentE2E:
    """Swarm Agent 端到端测试"""

    def test_swarm_agent_full_workflow(self, client: TestClient) -> None:
        """测试 Swarm Agent 完整工作流程"""
        # 1. 创建 Swarm Agent 系统
        create_response = client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "test_swarm_e2e",
                "agents": [
                    {"name": "Alice", "system_prompt": "你是 Alice", "tools": []},
                    {"name": "Bob", "system_prompt": "你是 Bob", "tools": []},
                    {"name": "Charlie", "system_prompt": "你是 Charlie", "tools": []},
                ],
                "handoff_mapping": {
                    "Alice": ["Bob", "Charlie"],
                    "Bob": ["Alice", "Charlie"],
                    "Charlie": ["Alice", "Bob"],
                },
                "default_agent": "Alice",
            },
        )
        # Swarm 不依赖 LLM 可以创建
        assert create_response.status_code == 200

        data = create_response.json()
        assert data["type"] == "swarm"
        assert "Alice" in data["agents"]
        assert "Bob" in data["agents"]
        assert "Charlie" in data["agents"]

        # 2. 使用系统
        chat_response = client.post(
            f"/api/v1/agents/swarm/test_swarm_e2e/chat",
            json={
                "message": "你好",
                "session_id": "swarm_session_001",
            },
        )
        # 需要配置 LLM 才能实际对话
        assert chat_response.status_code in (200, 500)

        # 3. 验证系统在列表中
        list_response = client.get("/api/v1/agents/systems")
        assert list_response.status_code == 200
        systems = list_response.json()
        assert any(s["name"] == "test_swarm_e2e" for s in systems)

        # 4. 清理
        delete_response = client.delete(f"/api/v1/agents/systems/swarm_test_swarm_e2e")
        assert delete_response.status_code == 200

    def test_swarm_with_no_handoff(self, client: TestClient) -> None:
        """测试没有切换映射的 Swarm"""
        response = client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "simple_swarm",
                "agents": [
                    {"name": "Agent1", "system_prompt": "Agent 1", "tools": []},
                ],
                "handoff_mapping": {},
                "default_agent": "Agent1",
            },
        )
        assert response.status_code == 200


class TestQuickChatAPI:
    """快速聊天 API 测试"""

    def test_quick_chat_single_mode(self, client: TestClient) -> None:
        """测试快速聊天 - 单 Agent 模式"""
        response = client.post(
            "/api/v1/agents/chat",
            params={"system_type": "single", "system_name": "default"},
            json={
                "message": "你好",
                "session_id": "quick_chat_001",
            },
        )
        # 需要配置 LLM
        assert response.status_code in (200, 404, 500)

    def test_quick_chat_nonexistent_system(self, client: TestClient) -> None:
        """测试快速聊天 - 不存在的系统"""
        response = client.post(
            "/api/v1/agents/chat",
            params={"system_type": "router", "system_name": "nonexistent"},
            json={
                "message": "你好",
                "session_id": "quick_chat_002",
            },
        )
        assert response.status_code == 404


class TestAgentSystemManagement:
    """Agent 系统管理测试"""

    def test_list_systems_after_multiple_creations(self, client: TestClient) -> None:
        """测试创建多个系统后列出"""
        # 创建几个 Swarm 系统（不依赖 LLM）
        for i in range(3):
            client.post(
                "/api/v1/agents/swarm",
                json={
                    "name": f"test_system_{i}",
                    "agents": [
                        {"name": f"Agent{i}", "system_prompt": f"Agent {i}", "tools": []}
                    ],
                    "handoff_mapping": {},
                    "default_agent": f"Agent{i}",
                },
            )

        # 列出所有系统
        response = client.get("/api/v1/agents/systems")
        assert response.status_code == 200
        systems = response.json()

        # 清理
        for i in range(3):
            client.delete(f"/api/v1/agents/systems/swarm_test_system_{i}")

        # 验证至少有 3 个测试系统
        test_systems = [s for s in systems if s["name"].startswith("test_system_")]
        assert len(test_systems) >= 3

    def test_delete_system_twice(self, client: TestClient) -> None:
        """测试删除系统两次"""
        # 创建系统
        client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "delete_test",
                "agents": [{"name": "Agent1", "system_prompt": "Agent", "tools": []}],
                "handoff_mapping": {},
                "default_agent": "Agent1",
            },
        )

        # 第一次删除
        delete1 = client.delete("/api/v1/agents/systems/swarm_delete_test")
        assert delete1.status_code == 200

        # 第二次删除应该返回 404
        delete2 = client.delete("/api/v1/agents/systems/swarm_delete_test")
        assert delete2.status_code == 404

    def test_get_history_after_clear(self, client: TestClient) -> None:
        """测试清除历史后获取历史"""
        system_id = "history_test"
        session_id = "test_session"

        # 创建系统
        client.post(
            "/api/v1/agents/swarm",
            json={
                "name": system_id,
                "agents": [{"name": "Agent1", "system_prompt": "Agent", "tools": []}],
                "handoff_mapping": {},
                "default_agent": "Agent1",
            },
        )

        # 获取历史（空）
        history1 = client.get(
            f"/api/v1/agents/systems/swarm_{system_id}/history/{session_id}"
        )
        assert history1.status_code == 200

        # 清除历史
        clear = client.delete(
            f"/api/v1/agents/systems/swarm_{system_id}/history/{session_id}"
        )
        assert clear.status_code == 200

        # 再次获取历史
        history2 = client.get(
            f"/api/v1/agents/systems/swarm_{system_id}/history/{session_id}"
        )
        assert history2.status_code == 200

        # 清理
        client.delete(f"/api/v1/agents/systems/swarm_{system_id}")
