"""Agent 工厂测试

测试 AgentFactory 的各种 Agent 创建功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.factory import (
    AgentFactory,
    AgentFactoryError,
    AgentType,
    create_agent,
)


class TestAgentFactory:
    """AgentFactory 测试"""

    def test_set_default_llm_service(self) -> None:
        """测试设置默认 LLM 服务"""
        mock_llm = MagicMock()
        AgentFactory.set_default_llm_service(mock_llm)
        assert AgentFactory._default_llm_service is mock_llm

    def test_set_default_checkpointer(self) -> None:
        """测试设置默认检查点"""
        mock_checkpointer = MagicMock()
        AgentFactory.set_default_checkpointer(mock_checkpointer)
        assert AgentFactory._default_checkpointer is mock_checkpointer

    @patch("app.agent.factory.get_llm_service")
    def test_create_chat_agent(self, mock_get_llm) -> None:
        """测试创建 Chat Agent"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.agent.factory.ChatGraph") as mock_graph_class:
            mock_graph = MagicMock()
            mock_graph_class.return_value = mock_graph

            result = AgentFactory.create_agent("chat")  # 使用字符串字面量

            assert result is mock_graph
            mock_graph.compile.assert_called_once()

    @patch("app.agent.factory.get_llm_service")
    def test_create_react_agent(self, mock_get_llm) -> None:
        """测试创建 ReAct Agent"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.agent.factory.create_react_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = AgentFactory.create_agent(
                "react",  # 使用字符串字面量
                tools=[MagicMock()],
            )

            assert result is mock_agent

    @patch("app.agent.factory.get_llm_service")
    def test_create_router_agent_without_agents(self, mock_get_llm) -> None:
        """测试创建 Router Agent 缺少 agents 参数"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with pytest.raises(AgentFactoryError, match="Router Agent 需要 agents 参数"):
            AgentFactory.create_agent("router")  # 使用字符串字面量

    @patch("app.agent.factory.get_llm_service")
    def test_create_router_agent_success(self, mock_get_llm) -> None:
        """测试成功创建 Router Agent"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        agents = {
            "agent1": MagicMock(_chat_node=MagicMock()),
            "agent2": MagicMock(_chat_node=MagicMock()),
        }

        result = AgentFactory.create_agent("router", agents=agents)  # 使用字符串字面量

        assert result is not None

    @patch("app.agent.factory.get_llm_service")
    def test_create_supervisor_agent_without_workers(self, mock_get_llm) -> None:
        """测试创建 Supervisor Agent 缺少 workers 参数"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with pytest.raises(AgentFactoryError, match="Supervisor Agent 需要 workers 参数"):
            AgentFactory.create_agent("supervisor")  # 使用字符串字面量

    @patch("app.agent.factory.get_llm_service")
    def test_create_supervisor_agent_success(self, mock_get_llm) -> None:
        """测试成功创建 Supervisor Agent"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        workers = {
            "worker1": MagicMock(_chat_node=MagicMock()),
            "worker2": MagicMock(_chat_node=MagicMock()),
        }

        result = AgentFactory.create_agent("supervisor", workers=workers)  # 使用字符串字面量

        assert result is not None

    @patch("app.agent.factory.get_llm_service")
    def test_create_handoff_agent_without_name(self, mock_get_llm) -> None:
        """测试创建 Handoff Agent 缺少 name 参数"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with pytest.raises(AgentFactoryError, match="Handoff Agent 需要 name 参数"):
            AgentFactory.create_agent("handoff")  # 使用字符串字面量

    @patch("app.agent.factory.get_llm_service")
    def test_create_handoff_agent_success(self, mock_get_llm) -> None:
        """测试成功创建 Handoff Agent"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        result = AgentFactory.create_agent(
            "handoff",  # 使用字符串字面量
            name="TestAgent",
            tools=[],
            handoff_targets=["Agent1"],
        )

        assert result is not None

    def test_create_agent_unknown_type(self) -> None:
        """测试创建未知类型的 Agent"""
        with pytest.raises(AgentFactoryError, match="未知的 Agent 类型"):
            AgentFactory.create_agent("unknown_type")  # type: ignore

    @patch("app.agent.factory.get_llm_service")
    def test_create_multi_agent_system(self, mock_get_llm) -> None:
        """测试创建多 Agent 系统"""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.agent.factory.create_multi_agent_system") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            result = AgentFactory.create_multi_agent_system("router")

            assert result is mock_graph


class TestCreateAgent:
    """create_agent 便捷函数测试"""

    @patch("app.agent.factory.AgentFactory.create_agent")
    def test_create_agent_calls_factory(self, mock_create) -> None:
        """测试便捷函数调用工厂方法"""
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        result = create_agent("chat")  # 使用字符串字面量

        assert result is mock_agent
        mock_create.assert_called_once()


@pytest.mark.parametrize("agent_type,required_kwarg", [
    ("router", "agents"),
    ("supervisor", "workers"),
    ("handoff", "name"),
])
def test_create_agent_missing_required_params(agent_type: str, required_kwarg: str) -> None:
    """参数化测试缺少必需参数"""
    with patch("app.agent.factory.get_llm_service"):
        with pytest.raises(AgentFactoryError, match=required_kwarg):
            AgentFactory.create_agent(agent_type)
