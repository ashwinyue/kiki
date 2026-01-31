"""LangGraph Agent 测试

测试 LangGraphAgent 管理类的功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.agent import (
    LangGraphAgent,
    create_agent,
    get_agent,
)


class TestLangGraphAgent:
    """LangGraphAgent 测试"""

    @patch("app.agent.agent.get_llm_service")
    def test_initialization(self, mock_get_llm) -> None:
        """测试初始化"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        agent = LangGraphAgent()

        assert agent._llm_service is mock_llm
        assert agent._graph is None
        assert agent._system_prompt is not None

    @patch("app.agent.agent.get_llm_service")
    def test_initialization_with_system_prompt(self, mock_get_llm) -> None:
        """测试带自定义提示词的初始化"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        custom_prompt = "You are a helpful assistant."
        agent = LangGraphAgent(system_prompt=custom_prompt)

        assert agent._system_prompt == custom_prompt

    @patch("app.agent.agent.get_llm_service")
    def test_initialization_with_checkpointer(self, mock_get_llm) -> None:
        """测试带检查点的初始化"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        mock_checkpointer = MagicMock()
        agent = LangGraphAgent(checkpointer=mock_checkpointer)

        assert agent._checkpointer is mock_checkpointer

    @patch("app.agent.agent.get_llm_service")
    def test_default_system_prompt(self, mock_get_llm) -> None:
        """测试默认系统提示词"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        agent = LangGraphAgent()

        prompt = agent._default_system_prompt()
        assert "AI 助手" in prompt
        assert "友好" in prompt

    @patch("app.agent.agent.get_llm_service")
    def test_get_graph_creates_graph(self, mock_get_llm) -> None:
        """测试获取图时创建新图"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        with patch("app.agent.agent.ChatGraph") as mock_graph_class:
            mock_graph = MagicMock()
            mock_graph_class.return_value = mock_graph

            agent = LangGraphAgent()
            graph = agent._get_graph()

            assert graph is mock_graph
            mock_graph_class.assert_called_once()

    @patch("app.agent.agent.get_llm_service")
    @patch("app.agent.agent._postgres_available", False)
    @patch("app.agent.agent._psycopg_pool_available", False)
    async def test_get_postgres_checkpointer_not_available(self, mock_get_llm) -> None:
        """测试 PostgreSQL 不可用时"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        agent = LangGraphAgent()

        checkpointer = await agent._get_postgres_checkpointer()

        assert checkpointer is None

    @patch("app.agent.agent.get_llm_service")
    async def test_close_without_connection_pool(self, mock_get_llm) -> None:
        """测试无连接池时关闭"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        agent = LangGraphAgent()

        # 应该不抛出异常
        await agent.close()

    @patch("app.agent.agent.get_llm_service")
    async def test_close_with_connection_pool(self, mock_get_llm) -> None:
        """测试关闭连接池"""
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        agent = LangGraphAgent()
        agent._connection_pool = mock_pool

        await agent.close()

        mock_pool.close.assert_called_once()


class TestCreateAgent:
    """create_agent 函数测试"""

    @patch("app.agent.agent.LangGraphAgent")
    def test_create_agent_default(self, mock_agent_class) -> None:
        """测试使用默认参数创建 Agent"""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        result = create_agent()

        assert result is mock_agent

    @patch("app.agent.agent.LangGraphAgent")
    def test_create_agent_with_system_prompt(self, mock_agent_class) -> None:
        """测试带系统提示词创建 Agent"""
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        custom_prompt = "You are a specialized assistant."
        result = create_agent(system_prompt=custom_prompt)

        assert result is mock_agent
        # 验证系统提示词参数被传递
        mock_agent_class.assert_called_once_with(
            llm_service=None,
            system_prompt=custom_prompt,
        )

    @patch("app.agent.agent.LangGraphAgent")
    def test_create_agent_with_llm_service(self, mock_agent_class) -> None:
        """测试带 LLM 服务创建 Agent"""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        result = create_agent(llm_service=mock_llm)

        assert result is mock_agent


class TestGetAgent:
    """get_agent 函数测试"""

    def test_get_agent_singleton(self) -> None:
        """测试单例模式"""
        # 重置全局变量
        import app.agent.agent
        app.agent.agent._agent = None

        with patch("app.agent.agent.LangGraphAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent1 = get_agent()
            agent2 = get_agent()

            # 验证返回同一个实例
            assert agent1 is agent2

    def test_get_agent_with_system_prompt(self) -> None:
        """测试带自定义提示词获取 Agent"""
        # 重置全局变量
        import app.agent.agent
        app.agent.agent._agent = None

        custom_prompt = "You are a specialized assistant."

        with patch("app.agent.agent.LangGraphAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent = get_agent(system_prompt=custom_prompt)

            assert agent is not None

    def test_get_agent_without_postgres_checkpointer(self) -> None:
        """测试不使用 PostgreSQL 检查点"""
        # 重置全局变量
        import app.agent.agent
        app.agent.agent._agent = None

        with patch("app.agent.agent.LangGraphAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent = get_agent(use_postgres_checkpointer=False)

            assert agent is not None


@pytest.mark.parametrize("has_llm,has_checkpointer,has_custom_prompt", [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
])
def test_agent_initialization_variations(has_llm: bool, has_checkpointer: bool, has_custom_prompt: bool) -> None:
    """参数化测试各种初始化组合"""
    with patch("app.agent.agent.get_llm_service") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.current_model = "gpt-4o"
        mock_get_llm.return_value = mock_llm

        llm_service = mock_llm if has_llm else None
        checkpointer = MagicMock() if has_checkpointer else None
        system_prompt = "Custom prompt" if has_custom_prompt else None

        agent = LangGraphAgent(
            llm_service=llm_service,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
        )

        assert agent is not None
