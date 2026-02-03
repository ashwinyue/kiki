"""Agent 状态 Pydantic 模型

提供 Pydantic 验证层，用于状态数据的验证、序列化和文档生成。

与 TypedDict 状态定义配合使用：
- TypedDict 用于 LangGraph StateGraph（框架要求）
- Pydantic 模型用于验证和文档（开发体验）

遵循原则：
1. 保持与 TypedDict 状态的字段对应
2. 添加字段验证器和约束
3. 提供与 TypedDict 的转换方法
4. 支持部分更新和默认值
"""

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== ChatState Pydantic 模型 ==============


class ChatStateModel(BaseModel):
    """ChatState 的 Pydantic 验证模型

    提供字段验证、默认值和类型转换。

    Attributes:
        messages: 消息列表（LangGraph 管理）
        user_id: 用户 ID
        session_id: 会话 ID（必须非空）
        tenant_id: 租户 ID
        iteration_count: 当前迭代次数（必须在 0-50 之间）
        max_iterations: 最大迭代次数（必须在 1-100 之间）
        error: 错误信息
    """

    # 消息历史
    messages: list[BaseMessage] = Field(default_factory=list, description="消息列表")

    # 用户和会话信息
    user_id: str | None = Field(default=None, description="用户 ID")
    session_id: str = Field(default="", description="会话 ID")
    tenant_id: int | None = Field(default=None, description="租户 ID")

    # 迭代控制
    iteration_count: int = Field(default=0, ge=0, le=50, description="当前迭代次数")
    max_iterations: int = Field(default=10, ge=1, le=100, description="最大迭代次数")

    # 错误处理
    error: str | None = Field(default=None, description="错误信息")

    @model_validator(mode="after")
    def validate_iteration_limits(self) -> "ChatStateModel":
        """验证迭代次数限制"""
        if self.iteration_count > self.max_iterations:
            logger.warning(
                "iteration_count_exceeds_max",
                iteration_count=self.iteration_count,
                max_iterations=self.max_iterations,
            )
            # 使用 object.__setattr__ 绕过 Pydantic 验证，避免递归
            object.__setattr__(self, "iteration_count", self.max_iterations)
        return self

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（与 TypedDict 兼容）"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatStateModel":
        """从字典创建模型（支持部分数据）

        Args:
            data: 状态字典（可能缺少某些字段）

        Returns:
            ChatStateModel 实例
        """
        # 提取已知字段
        filtered_data = {
            k: v
            for k, v in data.items()
            if k
            in {
                "messages",
                "user_id",
                "session_id",
                "tenant_id",
                "iteration_count",
                "max_iterations",
                "error",
            }
        }
        return cls(**filtered_data)

    def to_chat_state(self) -> dict[str, Any]:
        """转换为 ChatState 格式的字典

        这个方法返回的字典可以直接用作 LangGraph 的状态更新。
        """
        return self.to_dict()

    class Config:
        """Pydantic 配置"""

        # 使用枚举值而不是名称
        use_enum_values = True
        # 验证赋值（注意：这会在赋值时触发验证）
        validate_assignment = True


# ============== AgentState Pydantic 模型 ==============


class AgentStateModel(BaseModel):
    """AgentState 的 Pydantic 验证模型

    用于通用 Agent 状态的验证。
    """

    # 消息历史
    messages: list[BaseMessage] = Field(default_factory=list, description="消息列表")

    # 查询相关
    query: str = Field(default="", description="当前查询")
    rewrite_query: str | None = Field(default=None, description="重写后的查询")

    # 搜索和上下文
    search_results: list[Any] = Field(default_factory=list, description="搜索结果")
    context_str: str = Field(default="", description="构建的上下文")

    # 控制字段
    iteration_count: int = Field(default=0, ge=0, le=50, description="当前迭代次数")
    max_iterations: int = Field(default=10, ge=1, le=100, description="最大迭代次数")

    # 错误处理
    error: str | None = Field(default=None, description="错误信息")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStateModel":
        """从字典创建模型"""
        filtered_data = {
            k: v
            for k, v in data.items()
            if k
            in {
                "messages",
                "query",
                "rewrite_query",
                "search_results",
                "context_str",
                "iteration_count",
                "max_iterations",
                "error",
            }
        }
        return cls(**filtered_data)


# ============== ReActState Pydantic 模型 ==============


class ReActStateModel(BaseModel):
    """ReActState 的 Pydantic 验证模型

    用于 ReAct Agent 状态的验证。
    """

    # 消息历史
    messages: list[BaseMessage] = Field(default_factory=list, description="消息列表")

    # 工具调用
    tool_calls_to_execute: list[dict[str, Any]] = Field(
        default_factory=list, description="待执行的工具调用"
    )

    # 控制字段
    iteration_count: int = Field(default=0, ge=0, le=50, description="当前迭代次数")
    max_iterations: int = Field(default=10, ge=1, le=100, description="最大迭代次数")

    # 错误处理
    error: str | None = Field(default=None, description="错误信息")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


# ============== 状态验证器 ==============


class StateValidator:
    """状态验证器

    提供静态方法验证状态数据的有效性。
    """

    @staticmethod
    def validate_chat_state(data: dict[str, Any]) -> dict[str, Any]:
        """验证 ChatState 数据

        Args:
            data: 状态字典

        Returns:
            验证后的状态字典

        Raises:
            ValueError: 验证失败时
        """
        try:
            model = ChatStateModel.from_dict(data)
            return model.to_dict()
        except Exception as e:
            logger.error("chat_state_validation_failed", error=str(e))
            raise ValueError(f"ChatState 验证失败: {e}") from e

    @staticmethod
    def validate_agent_state(data: dict[str, Any]) -> dict[str, Any]:
        """验证 AgentState 数据"""
        try:
            model = AgentStateModel.from_dict(data)
            return model.to_dict()
        except Exception as e:
            logger.error("agent_state_validation_failed", error=str(e))
            raise ValueError(f"AgentState 验证失败: {e}") from e

    @staticmethod
    def validate_react_state(data: dict[str, Any]) -> dict[str, Any]:
        """验证 ReActState 数据"""
        try:
            model = ReActStateModel.from_dict(data)
            return model.to_dict()
        except Exception as e:
            logger.error("react_state_validation_failed", error=str(e))
            raise ValueError(f"ReActState 验证失败: {e}") from e

    @staticmethod
    def safe_validate_chat_state(data: dict[str, Any]) -> dict[str, Any] | None:
        """安全验证 ChatState 数据（不抛出异常）

        Args:
            data: 状态字典

        Returns:
            验证后的状态字典，失败返回 None
        """
        try:
            return StateValidator.validate_chat_state(data)
        except Exception:
            return None


# ============== 便捷函数 ==============


def validate_state_update(
    state: dict[str, Any], update: dict[str, Any], state_type: str = "chat"
) -> dict[str, Any]:
    """验证状态更新

    将当前状态和更新合并后验证。

    Args:
        state: 当前状态
        update: 状态更新
        state_type: 状态类型（chat, agent, react）

    Returns:
        验证后的完整状态
    """
    # 合并状态
    merged = {**state, **update}

    # 根据类型验证
    if state_type == "chat":
        return StateValidator.validate_chat_state(merged)
    elif state_type == "agent":
        return StateValidator.validate_agent_state(merged)
    elif state_type == "react":
        return StateValidator.validate_react_state(merged)
    else:
        raise ValueError(f"不支持的状态类型: {state_type}")


def increment_iteration_validated(state: dict[str, Any]) -> dict[str, Any]:
    """增加迭代计数（带验证）

    Args:
        state: 当前状态

    Returns:
        包含递增计数的状态更新
    """
    current = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    # 检查是否超过最大值
    if current >= max_iterations:
        logger.warning(
            "max_iterations_reached",
            iteration_count=current,
            max_iterations=max_iterations,
        )
        return {}

    return {"iteration_count": current + 1}


__all__ = [
    # Pydantic 模型
    "ChatStateModel",
    "AgentStateModel",
    "ReActStateModel",
    # 验证器
    "StateValidator",
    # 便捷函数
    "validate_state_update",
    "increment_iteration_validated",
]
