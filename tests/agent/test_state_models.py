"""Pydantic 状态模型测试

测试 Pydantic 状态模型的验证、转换和约束功能。
"""

import pytest
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state_models import (
    AgentStateModel,
    ChatStateModel,
    ReActStateModel,
    StateValidator,
    increment_iteration_validated,
    validate_state_update,
)


# ============== ChatStateModel 测试 ==============


class TestChatStateModel:
    """ChatStateModel 测试"""

    def test_create_default_state(self):
        """测试创建默认状态"""
        state = ChatStateModel()

        assert state.messages == []
        assert state.user_id is None
        assert state.session_id == ""
        assert state.tenant_id is None
        assert state.iteration_count == 0
        assert state.max_iterations == 10
        assert state.error is None
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)

    def test_create_state_with_values(self):
        """测试创建带值的状态"""
        messages = [HumanMessage(content="Hello")]
        state = ChatStateModel(
            messages=messages,
            user_id="user123",
            session_id="session456",
            tenant_id=1,
            iteration_count=5,
            max_iterations=20,
        )

        assert state.messages == messages
        assert state.user_id == "user123"
        assert state.session_id == "session456"
        assert state.tenant_id == 1
        assert state.iteration_count == 5
        assert state.max_iterations == 20

    def test_session_id_validation_empty_string(self):
        """测试 session_id 验证：空字符串应通过（有默认值）"""
        # 空字符串是允许的（有默认值）
        state = ChatStateModel(session_id="")
        assert state.session_id == ""

    def test_session_id_validation_none(self):
        """测试 session_id 验证：None 应使用默认值"""
        state = ChatStateModel(session_id=None)
        assert state.session_id == ""

    def test_session_id_validation_non_empty(self):
        """测试 session_id 验证：非空字符串应通过"""
        state = ChatStateModel(session_id="valid-session-id")
        assert state.session_id == "valid-session-id"

    def test_iteration_count_constraints(self):
        """测试迭代次数约束"""
        # 有效范围：0-50
        ChatStateModel(iteration_count=0)
        ChatStateModel(iteration_count=25)
        ChatStateModel(iteration_count=50)

        # 超出范围应引发错误
        with pytest.raises(ValueError):
            ChatStateModel(iteration_count=-1)

        with pytest.raises(ValueError):
            ChatStateModel(iteration_count=51)

    def test_max_iterations_constraints(self):
        """测试最大迭代次数约束"""
        # 有效范围：1-100
        ChatStateModel(max_iterations=1)
        ChatStateModel(max_iterations=50)
        ChatStateModel(max_iterations=100)

        # 超出范围应引发错误
        with pytest.raises(ValueError):
            ChatStateModel(max_iterations=0)

        with pytest.raises(ValueError):
            ChatStateModel(max_iterations=101)

    def test_iteration_auto_limit(self):
        """测试自动限制迭代次数到最大值"""
        # 当 iteration_count > max_iterations 时，应自动限制
        state = ChatStateModel(iteration_count=15, max_iterations=10)
        assert state.iteration_count == 10  # 自动限制到最大值

    def test_to_dict(self):
        """测试转换为字典"""
        messages = [HumanMessage(content="Hello")]
        state = ChatStateModel(
            messages=messages,
            user_id="user123",
            session_id="session456",
        )

        data = state.to_dict()

        # 验证字段存在
        assert "messages" in data
        assert "user_id" in data
        assert "session_id" in data
        assert "tenant_id" in data
        assert "iteration_count" in data
        assert "max_iterations" in data

        # 验证 Pydantic 特有字段被排除
        assert "created_at" not in data
        assert "updated_at" not in data

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
            "tenant_id": 1,
            "iteration_count": 5,
        }

        state = ChatStateModel.from_dict(data)

        assert state.messages == data["messages"]
        assert state.user_id == data["user_id"]
        assert state.session_id == data["session_id"]
        assert state.tenant_id == data["tenant_id"]
        assert state.iteration_count == data["iteration_count"]
        # 未指定的字段使用默认值
        assert state.max_iterations == 10

    def test_from_dict_with_extra_fields(self):
        """测试从包含额外字段的字典创建"""
        data = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "unknown_field": "should_be_ignored",  # 未知字段应被忽略
        }

        state = ChatStateModel.from_dict(data)

        assert state.user_id == "user123"
        # 未知字段不应导致错误


# ============== AgentStateModel 测试 ==============


class TestAgentStateModel:
    """AgentStateModel 测试"""

    def test_create_default_state(self):
        """测试创建默认状态"""
        state = AgentStateModel()

        assert state.messages == []
        assert state.query == ""
        assert state.rewrite_query is None
        assert state.search_results == []
        assert state.context_str == ""
        assert state.iteration_count == 0

    def test_create_state_with_values(self):
        """测试创建带值的状态"""
        messages = [HumanMessage(content="Search for Python")]
        state = AgentStateModel(
            messages=messages,
            query="Python tutorial",
            rewrite_query="Python programming tutorial",
            search_results=[{"title": "Python Guide"}],
            context_str="Python is a programming language",
        )

        assert state.messages == messages
        assert state.query == "Python tutorial"
        assert state.rewrite_query == "Python programming tutorial"
        assert state.search_results == [{"title": "Python Guide"}]
        assert state.context_str == "Python is a programming language"


# ============== ReActStateModel 测试 ==============


class TestReActStateModel:
    """ReActStateModel 测试"""

    def test_create_default_state(self):
        """测试创建默认状态"""
        state = ReActStateModel()

        assert state.messages == []
        assert state.tool_calls_to_execute == []
        assert state.iteration_count == 0

    def test_create_state_with_tool_calls(self):
        """测试创建带工具调用的状态"""
        messages = [AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "test"}}])]
        tool_calls = [
            {"name": "search", "arguments": {"query": "test"}},
            {"name": "calculate", "arguments": {"expression": "2+2"}},
        ]

        state = ReActStateModel(messages=messages, tool_calls_to_execute=tool_calls)

        assert state.messages == messages
        assert state.tool_calls_to_execute == tool_calls


# ============== StateValidator 测试 ==============


class TestStateValidator:
    """StateValidator 测试"""

    def test_validate_chat_state_success(self):
        """测试成功验证 ChatState"""
        data = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
            "iteration_count": 5,
        }

        result = StateValidator.validate_chat_state(data)

        assert result["user_id"] == "user123"
        assert result["session_id"] == "session456"
        assert result["iteration_count"] == 5

    def test_validate_chat_state_invalid_iteration(self):
        """测试验证无效的迭代次数"""
        data = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "iteration_count": -1,  # 无效值
        }

        with pytest.raises(ValueError, match="ChatState 验证失败"):
            StateValidator.validate_chat_state(data)

    def test_validate_agent_state_success(self):
        """测试成功验证 AgentState"""
        data = {
            "messages": [],
            "query": "test query",
            "context_str": "test context",
        }

        result = StateValidator.validate_agent_state(data)

        assert result["query"] == "test query"
        assert result["context_str"] == "test context"

    def test_safe_validate_chat_state_success(self):
        """测试安全验证成功"""
        data = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
        }

        result = StateValidator.safe_validate_chat_state(data)

        assert result is not None
        assert result["user_id"] == "user123"

    def test_safe_validate_chat_state_failure(self):
        """测试安全验证失败（不抛出异常）"""
        data = {
            "messages": [],
            "user_id": "user123",
            "session_id": "session456",
            "iteration_count": -1,  # 无效值
        }

        result = StateValidator.safe_validate_chat_state(data)

        # 失败时应返回 None 而不是抛出异常
        assert result is None


# ============== 便捷函数测试 ==============


class TestUtilityFunctions:
    """便捷函数测试"""

    def test_validate_state_update_chat(self):
        """测试验证状态更新 - ChatState"""
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
            "iteration_count": 5,
        }
        update = {"iteration_count": 6}

        result = validate_state_update(state, update, state_type="chat")

        assert result["iteration_count"] == 6
        assert result["user_id"] == "user123"

    def test_validate_state_update_agent(self):
        """测试验证状态更新 - AgentState"""
        state = {
            "messages": [],
            "query": "test",
            "iteration_count": 0,
        }
        update = {"query": "updated query"}

        result = validate_state_update(state, update, state_type="agent")

        assert result["query"] == "updated query"

    def test_validate_state_update_invalid_type(self):
        """测试验证状态更新 - 无效类型"""
        state = {"messages": []}
        update = {}

        with pytest.raises(ValueError, match="不支持的状态类型"):
            validate_state_update(state, update, state_type="invalid")

    def test_increment_iteration_validated(self):
        """测试带验证的增加迭代计数"""
        state = {
            "messages": [],
            "iteration_count": 5,
            "max_iterations": 10,
        }

        result = increment_iteration_validated(state)

        assert result["iteration_count"] == 6

    def test_increment_iteration_validated_max_reached(self):
        """测试达到最大迭代次数时"""
        state = {
            "messages": [],
            "iteration_count": 10,
            "max_iterations": 10,
        }

        result = increment_iteration_validated(state)

        # 达到最大值时应返回空字典
        assert result == {}

    def test_increment_iteration_validated_missing_fields(self):
        """测试缺少字段时的默认值"""
        state = {"messages": []}

        result = increment_iteration_validated(state)

        # 缺少 iteration_count 时默认为 0
        assert result["iteration_count"] == 1


# ============== 集成测试 ==============


class TestStateIntegration:
    """状态管理集成测试"""

    def test_complete_state_lifecycle(self):
        """测试完整的状态生命周期"""
        # 1. 创建状态
        data = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
        }

        # 2. 验证
        validated = StateValidator.validate_chat_state(data)
        assert validated is not None

        # 3. 转换为模型
        model = ChatStateModel.from_dict(validated)
        assert model.user_id == "user123"

        # 4. 转换回字典
        result = model.to_dict()
        assert result["user_id"] == "user123"

    def test_state_update_with_validation(self):
        """测试带验证的状态更新"""
        initial = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
            "iteration_count": 0,
            "max_iterations": 10,
        }

        # 多次更新
        for i in range(1, 6):
            update = increment_iteration_validated(initial)
            initial.update(update)

        assert initial["iteration_count"] == 5

    def test_preserve_meta_fields(self):
        """测试保留元字段"""
        from app.agent.state import preserve_state_meta_fields

        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_id": "user123",
            "session_id": "session456",
            "tenant_id": 1,
            "iteration_count": 5,
            "max_iterations": 10,
        }

        meta = preserve_state_meta_fields(state)

        assert meta["user_id"] == "user123"
        assert meta["session_id"] == "session456"
        assert meta["tenant_id"] == 1
        assert meta["iteration_count"] == 5
        assert meta["max_iterations"] == 10
        assert "messages" not in meta
