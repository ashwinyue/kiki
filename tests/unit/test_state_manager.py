"""状态管理器测试

测试 StateManager、StateUpdate 和 StateValidator 的功能。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.streaming.state_manager import (
    StateManager,
    StateUpdate,
    StateValidator,
    create_state_update,
    state_update,
)


class TestStateUpdate:
    """StateUpdate 测试"""

    def test_default_initialization(self) -> None:
        """测试默认初始化"""
        update = StateUpdate()

        assert update.messages == []
        assert update.iteration_delta == 0
        assert update.next_agent is None
        assert update.next_worker is None
        assert update.handoff_target is None
        assert update.metadata == {}

    def test_is_empty_when_new(self) -> None:
        """测试新建更新时为空"""
        update = StateUpdate()
        assert update.is_empty() is True

    def test_is_empty_with_messages(self) -> None:
        """测试有消息时不为空"""
        update = StateUpdate()
        update.messages = [AIMessage(content="test")]
        assert update.is_empty() is False

    def test_is_empty_with_iteration_delta(self) -> None:
        """测试有迭代增量时不为空"""
        update = StateUpdate()
        update.iteration_delta = 1
        assert update.is_empty() is False

    def test_to_dict_with_messages(self) -> None:
        """测试转换字典包含消息"""
        update = StateUpdate()
        update.messages = [AIMessage(content="test")]

        result = update.to_dict()
        assert "messages" in result
        assert len(result["messages"]) == 1

    def test_to_dict_with_next_agent(self) -> None:
        """测试转换字典包含下一个 Agent"""
        update = StateUpdate()
        update.next_agent = "worker_a"

        result = update.to_dict()
        assert result["_next_agent"] == "worker_a"

    def test_to_dict_with_handoff_target_none(self) -> None:
        """测试转换字典不包含 None 切换目标"""
        update = StateUpdate()
        update.handoff_target = None

        result = update.to_dict()
        # None 值不会被包含在结果中
        assert "_handoff_target" not in result


class TestStateValidator:
    """StateValidator 测试"""

    def test_validate_iteration_count_within_limit(self) -> None:
        """测试迭代次数在限制内"""
        state = AgentState({
            "iteration_count": 5,
            "max_iterations": 10,
        })
        assert StateValidator.validate_iteration_count(state) is True

    def test_validate_iteration_count_exceeded(self) -> None:
        """测试迭代次数超限"""
        state = AgentState({
            "iteration_count": 10,
            "max_iterations": 10,
        })
        assert StateValidator.validate_iteration_count(state) is False

    def test_validate_iteration_count_with_custom_max(self) -> None:
        """测试自定义最大迭代次数"""
        state = AgentState({
            "iteration_count": 5,
            "max_iterations": 100,
        })
        assert StateValidator.validate_iteration_count(state, max_iterations=3) is False

    def test_validate_required_fields_all_present(self) -> None:
        """测试所有必需字段存在且非空"""
        state = AgentState({
            "messages": [HumanMessage(content="test")],
            "user_id": "test_user",
        })
        assert StateValidator.validate_required_fields(state, ["messages", "user_id"]) is True

    def test_validate_required_fields_missing(self) -> None:
        """测试缺少必需字段"""
        state = AgentState({
            "messages": [],
        })
        assert StateValidator.validate_required_fields(state, ["messages", "user_id"]) is False

    def test_validate_message_count_within_limit(self) -> None:
        """测试消息数量在限制内"""
        state = AgentState({
            "messages": [HumanMessage(content=f"msg{i}") for i in range(10)],
        })
        assert StateValidator.validate_message_count(state, max_messages=100) is True

    def test_validate_message_count_exceeded(self) -> None:
        """测试消息数量超限"""
        state = AgentState({
            "messages": [HumanMessage(content=f"msg{i}") for i in range(1001)],
        })
        assert StateValidator.validate_message_count(state, max_messages=1000) is False


class TestStateManager:
    """StateManager 测试"""

    def test_initialization(self) -> None:
        """测试初始化"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        assert manager.state is state
        assert manager.iteration_count == 0
        assert manager.max_iterations == 10

    def test_initialization_with_custom_max_iterations(self) -> None:
        """测试自定义最大迭代次数"""
        state = AgentState({"messages": []})
        manager = StateManager(state, max_iterations=5)

        assert manager.max_iterations == 5

    def test_initialization_without_validation(self) -> None:
        """测试禁用验证"""
        state = AgentState({"messages": []})
        manager = StateManager(state, enable_validation=False)

        # 禁用验证时应该可以继续
        assert manager.can_continue() is True

    def test_add_message(self) -> None:
        """测试添加消息"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        msg = AIMessage(content="test")
        manager.add_message(msg)

        assert manager.has_updates() is True
        updates = manager.get_updates()
        assert "messages" in updates
        assert len(updates["messages"]) == 1

    def test_add_messages(self) -> None:
        """测试批量添加消息"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        messages = [AIMessage(content=f"msg{i}") for i in range(3)]
        manager.add_messages(messages)

        updates = manager.get_updates()
        assert len(updates["messages"]) == 3

    def test_increment_iteration(self) -> None:
        """测试递增迭代"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.increment_iteration()
        updates = manager.get_updates()
        assert updates["iteration_count"] == 1

    def test_increment_iteration_with_delta(self) -> None:
        """测试自定义增量递增"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.increment_iteration(delta=5)
        updates = manager.get_updates()
        assert updates["iteration_count"] == 5

    def test_set_next_agent(self) -> None:
        """测试设置下一个 Agent"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.set_next_agent("worker_a")
        updates = manager.get_updates()
        assert updates["_next_agent"] == "worker_a"

    def test_set_next_worker(self) -> None:
        """测试设置下一个 Worker"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.set_next_worker("worker_1")
        updates = manager.get_updates()
        assert updates["_next_worker"] == "worker_1"

    def test_set_handoff_target(self) -> None:
        """测试设置切换目标"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.set_handoff_target("Bob")
        updates = manager.get_updates()
        assert updates["_handoff_target"] == "Bob"

    def test_set_metadata(self) -> None:
        """测试设置元数据"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.set_metadata("key", "value")
        updates = manager.get_updates()
        assert updates["key"] == "value"

    def test_can_continue_when_within_limit(self) -> None:
        """测试在限制内可以继续"""
        state = AgentState({
            "messages": [],
            "iteration_count": 5,
            "max_iterations": 10,
        })
        manager = StateManager(state)

        assert manager.can_continue() is True
        assert manager.should_terminate() is False

    def test_can_continue_when_exceeded(self) -> None:
        """测试超过限制不能继续"""
        state = AgentState({
            "messages": [],
            "iteration_count": 10,
            "max_iterations": 10,
        })
        manager = StateManager(state)

        assert manager.can_continue() is False
        assert manager.should_terminate() is True

    def test_add_custom_validator(self) -> None:
        """测试添加自定义验证器"""
        state = AgentState({"messages": []})
        manager = StateManager(state, enable_validation=False)

        # 添加一个总是失败的验证器
        manager.add_validator(lambda s: False)

        assert manager.can_continue() is False

    def test_has_updates(self) -> None:
        """测试检查是否有更新"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        assert manager.has_updates() is False

        manager.add_message(AIMessage(content="test"))
        assert manager.has_updates() is True

    def test_reset_updates(self) -> None:
        """测试重置更新"""
        state = AgentState({"messages": []})
        manager = StateManager(state)

        manager.add_message(AIMessage(content="test"))
        assert manager.has_updates() is True

        manager.reset_updates()
        assert manager.has_updates() is False


class TestStateUpdateDecorator:
    """state_update 装饰器测试"""

    @pytest.mark.asyncio
    async def test_decorator_auto_increment(self) -> None:
        """测试自动递增装饰器"""
        state = AgentState({
            "messages": [],
            "iteration_count": 0,
        })

        @state_update(auto_increment=True)
        async def test_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="test")]}

        result = await test_node(state, {})
        assert "iteration_count" in result
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_decorator_with_messages(self) -> None:
        """测试装饰器处理消息"""
        state = AgentState({"messages": []})

        @state_update()
        async def test_node(state: AgentState, config) -> dict:
            return {"messages": [AIMessage(content="test")]}

        result = await test_node(state, {})
        assert "messages" in result
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_decorator_with_next_agent(self) -> None:
        """测试装饰器处理下一个 Agent"""
        state = AgentState({"messages": []})

        @state_update()
        async def test_node(state: AgentState, config) -> dict:
            return {"_next_agent": "worker_a"}

        result = await test_node(state, {})
        assert result["_next_agent"] == "worker_a"

    @pytest.mark.asyncio
    async def test_decorator_with_custom_metadata(self) -> None:
        """测试装饰器处理自定义元数据"""
        state = AgentState({"messages": []})

        @state_update()
        async def test_node(state: AgentState, config) -> dict:
            return {"custom_key": "custom_value"}

        result = await test_node(state, {})
        assert result["custom_key"] == "custom_value"


class TestCreateStateUpdate:
    """create_state_update 函数测试"""

    def test_create_with_messages(self) -> None:
        """测试创建带消息的更新"""
        update = create_state_update(
            messages=[AIMessage(content="test")]
        )
        assert "messages" in update
        assert len(update["messages"]) == 1

    def test_create_with_iteration_count(self) -> None:
        """测试创建带迭代计数的更新"""
        update = create_state_update(iteration_count=5)
        assert update["iteration_count"] == 5

    def test_create_with_next_agent(self) -> None:
        """测试创建带下一个 Agent 的更新"""
        update = create_state_update(_next_agent="worker_a")
        assert update["_next_agent"] == "worker_a"

    def test_create_with_multiple_fields(self) -> None:
        """测试创建带多个字段的更新"""
        update = create_state_update(
            messages=[AIMessage(content="test")],
            iteration_count=1,
            _next_agent="worker_a",
            _next_worker="worker_1",
            custom_key="custom_value",
        )
        assert "messages" in update
        assert update["iteration_count"] == 1
        assert update["_next_agent"] == "worker_a"
        assert update["_next_worker"] == "worker_1"
        assert update["custom_key"] == "custom_value"


@pytest.mark.parametrize("iteration_count,max_iterations,expected", [
    (0, 10, True),
    (5, 10, True),
    (9, 10, True),
    (10, 10, False),
    (11, 10, False),
])
def test_validate_iteration_count_variations(iteration_count: int, max_iterations: int, expected: bool) -> None:
    """参数化测试各种迭代计数场景"""
    state = AgentState({
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
    })
    assert StateValidator.validate_iteration_count(state) is expected
