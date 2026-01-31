"""多 Agent 系统测试

测试 Router、Supervisor 和 Handoff Agent 的路由和协作功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.multi_agent import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_handoff_tool,
    create_multi_agent_system,
    create_swarm,
)
from app.agent.multi_agent.router import DEFAULT_MAX_ITERATIONS
from app.agent.state import AgentState
from app.llm import LLMService


# 创建包含迭代计数器的最小状态
def _create_minimal_state(**kwargs) -> AgentState:
    """创建最小状态的辅助函数"""
    defaults = {
        "messages": [],
        "user_id": None,
        "session_id": None,
        "iteration_count": 0,
        "max_iterations": 10,
        "_next_agent": None,
        "_next_worker": None,
        "_handoff_target": None,
    }
    defaults.update(kwargs)
    return AgentState(defaults)


class TestRouterAgent:
    """RouterAgent 测试"""

    @pytest.fixture
    def mock_llm_service(self) -> LLMService:
        """模拟 LLM 服务"""
        mock_service = MagicMock(spec=LLMService)
        mock_llm = MagicMock()

        # 模拟结构化输出
        mock_decision = MagicMock()
        mock_decision.agent = "sales"
        mock_decision.reason = "用户询问销售相关问题"
        mock_decision.confidence = 0.95

        # 正确配置 mock 链式调用
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.ainvoke = AsyncMock(return_value=mock_decision)
        mock_service.get_llm.return_value = mock_llm

        return mock_service

    @pytest.fixture
    def mock_agents(self) -> dict:
        """模拟子 Agent"""
        async def mock_chat_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="Agent 响应")]}

        return {
            "sales": MagicMock(_chat_node=mock_chat_node),
            "support": MagicMock(_chat_node=mock_chat_node),
            "general": MagicMock(_chat_node=mock_chat_node),
        }

    def test_router_initialization(self, mock_llm_service, mock_agents) -> None:
        """测试路由 Agent 初始化"""
        router = RouterAgent(mock_llm_service, mock_agents)

        assert router._agents == mock_agents
        assert len(router._agent_names) == 3
        assert "sales" in router._agent_names
        assert router._max_iterations == DEFAULT_MAX_ITERATIONS

    def test_router_initialization_with_custom_max_iterations(self, mock_llm_service, mock_agents) -> None:
        """测试自定义最大迭代次数"""
        router = RouterAgent(mock_llm_service, mock_agents, max_iterations=5)
        assert router._max_iterations == 5

    @pytest.mark.asyncio
    async def test_route_node_decision(self, mock_llm_service, mock_agents) -> None:
        """测试路由节点决策"""
        router = RouterAgent(mock_llm_service, mock_agents)

        state = _create_minimal_state(
            messages=[HumanMessage(content="我想了解产品价格")]
        )

        config = MagicMock()

        result = await router._route_node(state, config)

        assert "_next_agent" in result
        assert result["_next_agent"] in mock_agents
        assert "messages" in result
        assert result.get("iteration_count") == 1  # 验证迭代计数递增

    @pytest.mark.asyncio
    async def test_route_edge_function(self, mock_llm_service, mock_agents) -> None:
        """测试路由边函数"""
        router = RouterAgent(mock_llm_service, mock_agents)

        state_with_target = _create_minimal_state(_next_agent="sales")

        assert router._route_edge(state_with_target) == "sales"

        state_without_target = _create_minimal_state()
        # 无目标时返回第一个可用 Agent
        assert router._route_edge(state_without_target) in mock_agents

    @pytest.mark.asyncio
    async def test_route_edge_max_iterations(self, mock_llm_service, mock_agents) -> None:
        """测试达到最大迭代次数时强制结束"""
        router = RouterAgent(mock_llm_service, mock_agents, max_iterations=5)

        # 达到最大迭代次数
        state_max_iterations = _create_minimal_state(
            _next_agent="sales",
            iteration_count=5
        )

        from langgraph.graph import END
        assert router._route_edge(state_max_iterations) == END

    def test_compile_router_graph(self, mock_llm_service, mock_agents) -> None:
        """测试编译路由图"""
        router = RouterAgent(mock_llm_service, mock_agents)
        graph = router.compile()

        assert graph is not None


class TestSupervisorAgent:
    """SupervisorAgent 测试"""

    @pytest.fixture
    def mock_llm_service(self) -> LLMService:
        """模拟 LLM 服务"""
        mock_service = MagicMock(spec=LLMService)
        mock_llm = MagicMock()

        # 模拟结构化输出
        mock_decision = MagicMock()
        mock_decision.next = "worker1"
        mock_decision.status = "working"
        mock_decision.message = "分配给 worker1"

        # 正确配置 mock 链式调用
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm.ainvoke = AsyncMock(return_value=mock_decision)
        mock_service.get_llm.return_value = mock_llm

        return mock_service

    @pytest.fixture
    def mock_workers(self) -> dict:
        """模拟 Worker Agent"""
        async def mock_chat_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="Worker 完成")]}

        return {
            "worker1": MagicMock(_chat_node=mock_chat_node),
            "worker2": MagicMock(_chat_node=mock_chat_node),
        }

    def test_supervisor_initialization(self, mock_llm_service, mock_workers) -> None:
        """测试监督 Agent 初始化"""
        supervisor = SupervisorAgent(mock_llm_service, mock_workers)

        assert supervisor._workers == mock_workers
        assert len(supervisor._worker_names) == 2
        assert SupervisorAgent.TASK_DONE == "__done__"
        assert supervisor._max_iterations == DEFAULT_MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_supervise_node_working(self, mock_llm_service, mock_workers) -> None:
        """测试监督节点决策 - 工作中状态"""
        from app.agent.schemas import SupervisorDecision

        mock_decision = SupervisorDecision(
            next="worker1",
            status="working",
            message="分配给 worker1"
        )

        supervisor = SupervisorAgent(mock_llm_service, mock_workers)

        state = _create_minimal_state(
            messages=[AIMessage(content="开始任务")]
        )

        config = MagicMock()

        # 使用 patch 模拟链式调用
        with patch('app.agent.multi_agent.ChatPromptTemplate') as mock_prompt_template:
            # 配置 mock 返回正确的决策
            mock_prompt_template.from_messages.return_value.__or__.return_value.ainvoke = AsyncMock(
                return_value=mock_decision
            )

            result = await supervisor._supervise_node(state, config)

            assert "_next_worker" in result
            assert result["_next_worker"] == "worker1"
            assert "messages" in result
            assert result.get("iteration_count") == 1  # 验证迭代计数递增

    @pytest.mark.asyncio
    async def test_supervise_edge_function(self, mock_llm_service, mock_workers) -> None:
        """测试监督边函数"""
        supervisor = SupervisorAgent(mock_llm_service, mock_workers)

        # 测试正常工作状态
        state_working = _create_minimal_state(_next_worker="worker1")
        assert supervisor._supervise_edge(state_working) == "worker1"

        # 测试完成状态
        state_done = _create_minimal_state(_next_worker=SupervisorAgent.TASK_DONE)
        assert supervisor._supervise_edge(state_done) == "__end__"

    @pytest.mark.asyncio
    async def test_supervise_edge_max_iterations(self, mock_llm_service, mock_workers) -> None:
        """测试达到最大迭代次数时强制结束"""
        supervisor = SupervisorAgent(mock_llm_service, mock_workers, max_iterations=5)

        # 达到最大迭代次数
        state_max_iterations = _create_minimal_state(
            _next_worker="worker1",
            iteration_count=5
        )

        from langgraph.graph import END
        assert supervisor._supervise_edge(state_max_iterations) == END


class TestHandoffAgent:
    """HandoffAgent 测试"""

    @pytest.fixture
    def mock_llm_service(self) -> LLMService:
        """模拟 LLM 服务"""
        mock_service = MagicMock(spec=LLMService)
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="Agent 响应", tool_calls=[])
        )
        mock_service.get_llm.return_value = mock_llm
        return mock_service

    def test_handoff_agent_initialization(self, mock_llm_service) -> None:
        """测试切换 Agent 初始化"""
        agent = HandoffAgent(
            name="Alice",
            llm_service=mock_llm_service,
            handoff_targets=["Bob", "Charlie"],
        )

        assert agent.name == "Alice"
        assert len(agent._handoff_targets) == 2
        assert "Bob" in agent._handoff_targets
        assert len(agent._handoff_tools) == 2

    def test_get_handoff_targets(self, mock_llm_service) -> None:
        """测试获取切换目标"""
        agent = HandoffAgent(
            name="Alice",
            llm_service=mock_llm_service,
            handoff_targets=["Bob", "Charlie"],
        )

        targets = agent.get_handoff_targets()
        assert targets == ["Bob", "Charlie"]

    def test_create_handoff_tool(self) -> None:
        """测试创建切换工具"""
        tool = create_handoff_tool("Bob", "切换到 Bob")

        assert "transfer_to_bob" in tool.name
        assert tool._handoff_target == "Bob"


class TestCreateSwarm:
    """create_swarm 测试"""

    @pytest.fixture
    def mock_llm_service(self) -> LLMService:
        """模拟 LLM 服务"""
        mock_service = MagicMock(spec=LLMService)
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="响应", tool_calls=[])
        )
        mock_service.get_llm.return_value = mock_llm
        return mock_service

    def test_create_swarm(self, mock_llm_service) -> None:
        """测试创建 Agent Swarm"""
        alice = HandoffAgent(
            name="Alice",
            llm_service=mock_llm_service,
            handoff_targets=["Bob"],
        )
        bob = HandoffAgent(
            name="Bob",
            llm_service=mock_llm_service,
            handoff_targets=["Alice"],
        )

        swarm = create_swarm(agents=[alice, bob], default_agent="Alice")

        assert swarm is not None


class TestCreateMultiAgentSystem:
    """create_multi_agent_system 测试"""

    @pytest.fixture
    def mock_llm_service(self) -> LLMService:
        """模拟 LLM 服务"""
        mock_service = MagicMock(spec=LLMService)
        mock_llm = MagicMock()

        mock_decision = MagicMock()
        mock_decision.agent = "agent1"
        mock_decision.reason = "路由原因"
        mock_decision.confidence = 0.9

        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_decision
        )
        mock_service.get_llm.return_value = mock_llm
        return mock_service

    def test_create_router_system(self, mock_llm_service) -> None:
        """测试创建路由系统"""
        async def mock_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="响应")]}

        agents = {
            "agent1": MagicMock(_chat_node=mock_node),
            "agent2": MagicMock(_chat_node=mock_node),
        }

        graph = create_multi_agent_system(
            mode="router",
            llm_service=mock_llm_service,
            agents=agents,
        )

        assert graph is not None

    def test_create_supervisor_system(self, mock_llm_service) -> None:
        """测试创建监督系统"""
        async def mock_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="响应")]}

        workers = {
            "worker1": MagicMock(_chat_node=mock_node),
            "worker2": MagicMock(_chat_node=mock_node),
        }

        graph = create_multi_agent_system(
            mode="supervisor",
            llm_service=mock_llm_service,
            workers=workers,
        )

        assert graph is not None

    def test_invalid_mode(self, mock_llm_service) -> None:
        """测试无效模式"""
        with pytest.raises(ValueError, match="Unknown multi-agent mode"):
            create_multi_agent_system(
                mode="invalid",
                llm_service=mock_llm_service,
            )
