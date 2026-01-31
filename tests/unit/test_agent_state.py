"""Agent 状态管理测试

测试 AgentState 的创建、更新和消息管理功能。
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import (
    AgentState,
    _add_iteration,
    create_initial_state,
    create_state_from_input,
)


class TestAgentState:
    """AgentState 测试"""

    def test_create_initial_state_empty(self) -> None:
        """测试创建空的初始状态"""
        state = create_initial_state()

        assert state["messages"] == []
        assert state["user_id"] is None
        assert state["session_id"] is None
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 10
        assert state.get("_next_agent") is None
        assert state.get("_next_worker") is None
        assert state.get("_handoff_target") is None

    def test_create_initial_state_with_messages(self) -> None:
        """测试创建带消息的初始状态"""
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮助你的？"),
        ]
        state = create_initial_state(messages=messages)

        assert len(state["messages"]) == 2
        assert state["messages"][0].content == "你好"
        assert state["messages"][1].content == "你好！有什么可以帮助你的？"
        assert state["iteration_count"] == 0

    def test_create_initial_state_with_metadata(self) -> None:
        """测试创建带元数据的初始状态"""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            max_iterations=20,
        )

        assert state["user_id"] == "user-123"
        assert state["session_id"] == "session-456"
        assert state["max_iterations"] == 20
        assert state["iteration_count"] == 0

    def test_create_state_from_input(self) -> None:
        """测试从用户输入创建状态"""
        state = create_state_from_input(
            input_text="今天天气怎么样？",
            user_id="user-123",
            session_id="session-456",
        )

        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "今天天气怎么样？"
        assert state["user_id"] == "user-123"
        assert state["session_id"] == "session-456"
        assert state["iteration_count"] == 0

    def test_agent_state_structure(self) -> None:
        """测试 AgentState 结构完整性"""
        state: AgentState = {
            "messages": [
                SystemMessage(content="你是一个助手"),
                HumanMessage(content="你好"),
            ],
            "user_id": "user-123",
            "session_id": "session-456",
            "iteration_count": 0,
            "max_iterations": 10,
            "_next_agent": None,
            "_next_worker": None,
            "_handoff_target": None,
        }

        assert len(state["messages"]) == 2
        assert state["user_id"] == "user-123"
        assert state["session_id"] == "session-456"

    def test_add_iteration_reducer(self) -> None:
        """测试迭代计数累加器"""
        # 正常累加
        assert _add_iteration(0, 1) == 1
        assert _add_iteration(5, 1) == 6
        assert _add_iteration(5, 3) == 8

        # 处理 None 值
        assert _add_iteration(None, 1) == 1
        assert _add_iteration(5, None) == 5
        assert _add_iteration(None, None) == 0
