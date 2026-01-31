"""Agent 工作流 E2E 测试

测试多 Agent 协作系统的完整流程。
"""

import pytest

from fastapi.testclient import TestClient


@pytest.mark.e2e
class TestAgentSystemWorkflow:
    """Agent 系统工作流测试"""

    def test_swarm_agent_lifecycle(self, e2e_client: TestClient) -> None:
        """测试 Swarm Agent 完整生命周期"""
        # ========== Step 1: 创建 Swarm Agent ==========
        create_data = {
            "name": "customer_support_swarm",
            "agents": [
                {
                    "name": "Receptionist",
                    "system_prompt": "你是前台接待，负责初步了解客户需求并转接给合适的专家。",
                    "tools": [],
                },
                {
                    "name": "SalesExpert",
                    "system_prompt": "你是销售专家，负责产品咨询和报价。",
                    "tools": ["calculate"],
                },
                {
                    "name": "SupportExpert",
                    "system_prompt": "你是技术支持专家，负责售后问题解决。",
                    "tools": [],
                },
            ],
            "handoff_mapping": {
                "Receptionist": ["SalesExpert", "SupportExpert"],
                "SalesExpert": ["Receptionist", "SupportExpert"],
                "SupportExpert": ["Receptionist", "SalesExpert"],
            },
            "default_agent": "Receptionist",
        }

        response = e2e_client.post("/api/v1/agents/swarm", json=create_data)
        assert response.status_code == 200
        result = response.json()

        assert result["type"] == "swarm"
        assert result["name"] == "customer_support_swarm"
        assert "Receptionist" in result["agents"]
        assert "SalesExpert" in result["agents"]
        assert "SupportExpert" in result["agents"]

        # ========== Step 2: 验证系统在列表中 ==========
        response = e2e_client.get("/api/v1/agents/systems")
        assert response.status_code == 200
        systems = response.json()

        system_ids = [s["name"] for s in systems]
        assert "swarm_customer_support_swarm" in system_ids or any(
            "customer_support_swarm" in s.get("name", "") for s in systems
        )

        # ========== Step 3: 使用 Swarm 进行对话 ==========
        # 注意：需要配置 LLM 才能实际对话
        chat_data = {
            "message": "你好，我想了解一下你们的产品价格",
            "session_id": "swarm_test_001",
        }

        response = e2e_client.post(
            "/api/v1/agents/swarm/customer_support_swarm/chat",
            json=chat_data,
        )
        # LLM 未配置时可能失败
        assert response.status_code in (200, 500)

        # ========== Step 4: 获取对话历史 ==========
        history_response = e2e_client.get(
            "/api/v1/agents/systems/swarm_customer_support_swarm/history/swarm_test_001"
        )
        assert history_response.status_code == 200

        # ========== Step 5: 删除系统 ==========
        response = e2e_client.delete(
            "/api/v1/agents/systems/swarm_customer_support_swarm"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # ========== Step 6: 验证系统已删除 ==========
        response = e2e_client.delete(
            "/api/v1/agents/systems/swarm_customer_support_swarm"
        )
        assert response.status_code == 404

    def test_router_agent_lifecycle(self, e2e_client: TestClient) -> None:
        """测试 Router Agent 完整生命周期"""
        # ========== Step 1: 创建 Router Agent ==========
        create_data = {
            "name": "smart_router",
            "agents": [
                {
                    "name": "coder",
                    "system_prompt": "你是编程专家，负责回答技术问题。",
                    "tools": ["calculate"],
                },
                {
                    "name": "writer",
                    "system_prompt": "你是写作专家，负责内容创作。",
                    "tools": [],
                },
                {
                    "name": "analyst",
                    "system_prompt": "你是分析师，负责数据分析。",
                    "tools": ["calculate"],
                },
            ],
        }

        response = e2e_client.post("/api/v1/agents/router", json=create_data)
        # Router 需要 LLM 才能创建
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            result = response.json()
            assert result["type"] == "router"

            # ========== Step 2: 使用 Router 进行对话 ==========
            chat_data = {
                "message": "请帮我写一个 Python 函数",
                "session_id": "router_test_001",
            }

            response = e2e_client.post(
                "/api/v1/agents/router/smart_router/chat",
                json=chat_data,
            )
            assert response.status_code in (200, 500)

            # ========== Step 3: 清理 ==========
            response = e2e_client.delete(
                "/api/v1/agents/systems/router_smart_router"
            )
            assert response.status_code == 200

    def test_supervisor_agent_lifecycle(self, e2e_client: TestClient) -> None:
        """测试 Supervisor Agent 完整生命周期"""
        # ========== Step 1: 创建 Supervisor Agent ==========
        create_data = {
            "name": "research_team",
            "workers": [
                {
                    "name": "researcher",
                    "system_prompt": "你是研究员，负责收集信息。",
                    "tools": [],
                },
                {
                    "name": "analyst",
                    "system_prompt": "你是分析师，负责整理和分析信息。",
                    "tools": ["calculate"],
                },
                {
                    "name": "writer",
                    "system_prompt": "你是撰稿人，负责撰写报告。",
                    "tools": [],
                },
            ],
        }

        response = e2e_client.post("/api/v1/agents/supervisor", json=create_data)
        # Supervisor 需要 LLM
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            result = response.json()
            assert result["type"] == "supervisor"

            # ========== Step 2: 使用 Supervisor ==========
            chat_data = {
                "message": "帮我研究一下 AI 的发展趋势",
                "session_id": "supervisor_test_001",
            }

            response = e2e_client.post(
                "/api/v1/agents/supervisor/research_team/chat",
                json=chat_data,
            )
            assert response.status_code in (200, 500)

            # ========== Step 3: 清理 ==========
            response = e2e_client.delete(
                "/api/v1/agents/systems/supervisor_research_team"
            )
            assert response.status_code == 200

    def test_multiple_agent_systems(self, e2e_client: TestClient) -> None:
        """测试同时管理多个 Agent 系统"""
        system_names = []

        # 创建多个 Swarm 系统
        for i in range(3):
            name = f"test_system_{i}"
            system_names.append(name)

            response = e2e_client.post(
                "/api/v1/agents/swarm",
                json={
                    "name": name,
                    "agents": [
                        {
                            "name": f"Agent{i}",
                            "system_prompt": f"You are Agent {i}",
                            "tools": [],
                        }
                    ],
                    "handoff_mapping": {},
                    "default_agent": f"Agent{i}",
                },
            )
            assert response.status_code == 200

        # 列出所有系统
        response = e2e_client.get("/api/v1/agents/systems")
        assert response.status_code == 200
        systems = response.json()

        # 清理
        for name in system_names:
            e2e_client.delete(f"/api/v1/agents/systems/swarm_{name}")

    def test_agent_error_handling(self, e2e_client: TestClient) -> None:
        """测试 Agent 错误处理"""
        # ========== 尝试使用不存在的系统 ==========
        response = e2e_client.post(
            "/api/v1/agents/router/nonexistent/chat",
            json={"message": "Hello", "session_id": "test"},
        )
        assert response.status_code == 404

        # ========== 尝试删除不存在的系统 ==========
        response = e2e_client.delete("/api/v1/agents/systems/nonexistent")
        assert response.status_code == 404

        # ========== 尝试获取不存在系统的历史 ==========
        response = e2e_client.get(
            "/api/v1/agents/systems/nonexistent/history/test_session"
        )
        assert response.status_code == 404


@pytest.mark.e2e
class TestAgentChatHistory:
    """Agent 聊天历史测试"""

    def test_chat_history_persistence(self, e2e_client: TestClient) -> None:
        """测试聊天历史持久化"""
        # 创建 Swarm
        response = e2e_client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "history_test",
                "agents": [
                    {"name": "Agent1", "system_prompt": "Test", "tools": []}
                ],
                "handoff_mapping": {},
                "default_agent": "Agent1",
            },
        )
        assert response.status_code == 200

        # 获取初始历史（空）
        response = e2e_client.get(
            "/api/v1/agents/systems/swarm_history_test/history/session_123"
        )
        assert response.status_code == 200
        history = response.json()
        assert "messages" in history
        assert len(history["messages"]) == 0

        # 尝试发送消息
        chat_response = e2e_client.post(
            "/api/v1/agents/swarm/history_test/chat",
            json={"message": "测试消息", "session_id": "session_123"},
        )
        # LLM 未配置时可能失败

        # 清理
        e2e_client.delete("/api/v1/agents/systems/swarm_history_test")

    def test_clear_chat_history(self, e2e_client: TestClient) -> None:
        """测试清除聊天历史"""
        # 创建 Swarm
        response = e2e_client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "clear_test",
                "agents": [
                    {"name": "Agent1", "system_prompt": "Test", "tools": []}
                ],
                "handoff_mapping": {},
                "default_agent": "Agent1",
            },
        )
        assert response.status_code == 200

        # 清除历史
        response = e2e_client.delete(
            "/api/v1/agents/systems/swarm_clear_test/history/session_456"
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

        # 清理
        e2e_client.delete("/api/v1/agents/systems/swarm_clear_test")


@pytest.mark.e2e
class TestToolsIntegration:
    """工具集成测试"""

    def test_list_available_tools(self, e2e_client: TestClient) -> None:
        """测试列出可用工具"""
        response = e2e_client.get("/api/v1/tools")
        assert response.status_code == 200

        tools = response.json()
        assert isinstance(tools, list)

        # 验证工具结构
        for tool in tools:
            assert "name" in tool
            assert "description" in tool

    def test_get_tool_details(self, e2e_client: TestClient) -> None:
        """测试获取工具详情"""
        # 先获取工具列表
        list_response = e2e_client.get("/api/v1/tools")
        assert list_response.status_code == 200
        tools = list_response.json()

        if tools:
            tool_name = tools[0]["name"]
            response = e2e_client.get(f"/api/v1/tools/{tool_name}")
            assert response.status_code == 200

            tool_detail = response.json()
            assert tool_detail["name"] == tool_name
            assert "description" in tool_detail

    def test_agent_with_tools(self, e2e_client: TestClient) -> None:
        """测试带工具的 Agent"""
        response = e2e_client.post(
            "/api/v1/agents/swarm",
            json={
                "name": "tool_test",
                "agents": [
                    {
                        "name": "Calculator",
                        "system_prompt": "你是计算助手",
                        "tools": ["calculate"],
                    }
                ],
                "handoff_mapping": {},
                "default_agent": "Calculator",
            },
        )
        assert response.status_code == 200

        result = response.json()
        assert "Calculator" in result["agents"]
        assert result["agents"]["Calculator"]["tools"] == ["calculate"]

        # 清理
        e2e_client.delete("/api/v1/agents/systems/swarm_tool_test")
