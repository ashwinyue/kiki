"""状态管理器

集中管理 Agent 状态转换，提供统一的更新接口和验证机制。
符合 langgraph-agents 最佳实践。

使用示例:
```python
from app.agent.state_manager import StateManager, state_update

# 在节点中使用
async def my_node(state: AgentState, config: RunnableConfig) -> dict:
    manager = StateManager(state)
    manager.add_message(AIMessage(content="..."))
    manager.increment_iteration()
    manager.set_next_agent("worker_a")

    # 验证状态
    if not manager.can_continue():
        return {"_should_terminate": True}

    return manager.get_updates()
```
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from app.agent.state import AgentState
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StateUpdate:
    """状态更新数据类

    Attributes:
        messages: 要添加的消息
        iteration_delta: 迭代计数的增量
        next_agent: 下一个 Agent
        next_worker: 下一个 Worker
        handoff_target: 切换目标
        metadata: 额外元数据
    """

    messages: list[BaseMessage] = field(default_factory=list)
    iteration_delta: int = 0
    next_agent: str | None = None
    next_worker: str | None = None
    handoff_target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 LangGraph 状态更新字典"""
        updates: dict[str, Any] = {}

        if self.messages:
            updates["messages"] = self.messages

        if self.iteration_delta != 0:
            updates["iteration_count"] = self.iteration_delta

        if self.next_agent:
            updates["_next_agent"] = self.next_agent

        if self.next_worker:
            updates["_next_worker"] = self.next_worker

        if self.handoff_target is not None:  # 允许 None 值（表示取消切换）
            updates["_handoff_target"] = self.handoff_target

        if self.metadata:
            updates.update(self.metadata)

        return updates

    def is_empty(self) -> bool:
        """检查是否有任何更新"""
        return not any(
            [
                self.messages,
                self.iteration_delta != 0,
                self.next_agent,
                self.next_worker,
                self.handoff_target is not None,
                self.metadata,
            ]
        )


class StateValidator:
    """状态验证器

    提供状态验证规则和检查函数。
    """

    @staticmethod
    def validate_iteration_count(state: AgentState, max_iterations: int | None = None) -> bool:
        """验证迭代次数是否在允许范围内

        Args:
            state: 当前状态
            max_iterations: 最大迭代次数（默认使用 state 中的配置）

        Returns:
            是否可以继续
        """
        max_iter = max_iterations or state.get("max_iterations", 10)
        current = state.get("iteration_count", 0)

        can_continue = current < max_iter
        if not can_continue:
            logger.warning(
                "max_iterations_exceeded",
                current=current,
                max=max_iter,
            )

        return can_continue

    @staticmethod
    def validate_required_fields(state: AgentState, required_fields: list[str]) -> bool:
        """验证必需字段是否存在

        Args:
            state: 当前状态
            required_fields: 必需字段列表

        Returns:
            是否通过验证
        """
        missing = [field for field in required_fields if not state.get(field)]
        if missing:
            logger.warning("missing_required_fields", fields=missing)
            return False
        return True

    @staticmethod
    def validate_message_count(state: AgentState, max_messages: int = 1000) -> bool:
        """验证消息数量是否超限

        Args:
            state: 当前状态
            max_messages: 最大消息数量

        Returns:
            是否通过验证
        """
        message_count = len(state.get("messages", []))
        if message_count > max_messages:
            logger.warning(
                "message_count_exceeded",
                count=message_count,
                max=max_messages,
            )
            return False
        return True


class StateManager:
    """状态管理器

    集中管理状态转换，提供类型安全的更新方法。
    符合 SOLID 原则中的单一职责原则。
    """

    def __init__(
        self,
        state: AgentState,
        max_iterations: int | None = None,
        enable_validation: bool = True,
    ) -> None:
        """初始化状态管理器

        Args:
            state: 当前状态
            max_iterations: 最大迭代次数（覆盖状态中的配置）
            enable_validation: 是否启用状态验证
        """
        self._state = state
        self._max_iterations = max_iterations
        self._enable_validation = enable_validation
        self._update = StateUpdate()
        self._validators: list[Callable[[AgentState], bool]] = []

        # 添加默认验证器
        if enable_validation:
            self.add_validator(lambda s: StateValidator.validate_iteration_count(s, max_iterations))

    @property
    def state(self) -> AgentState:
        """获取当前状态"""
        return self._state

    @property
    def iteration_count(self) -> int:
        """获取当前迭代次数"""
        return self._state.get("iteration_count", 0)

    @property
    def max_iterations(self) -> int:
        """获取最大迭代次数"""
        return self._max_iterations or self._state.get("max_iterations", 10)

    def add_validator(self, validator: Callable[[AgentState], bool]) -> None:
        """添加自定义验证器

        Args:
            validator: 验证函数，返回 True 表示通过
        """
        self._validators.append(validator)

    def add_message(self, message: BaseMessage) -> None:
        """添加消息到状态

        Args:
            message: 要添加的消息
        """
        self._update.messages.append(message)
        logger.debug("message_added", type=message.type, content_preview=str(message.content)[:50])

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """批量添加消息

        Args:
            messages: 要添加的消息列表
        """
        self._update.messages.extend(messages)
        logger.debug("messages_added", count=len(messages))

    def increment_iteration(self, delta: int = 1) -> None:
        """递增迭代计数

        Args:
            delta: 增量值
        """
        self._update.iteration_delta = delta
        logger.debug("iteration_incremented", delta=delta)

    def set_next_agent(self, agent_name: str) -> None:
        """设置下一个路由目标 Agent

        Args:
            agent_name: 目标 Agent 名称
        """
        self._update.next_agent = agent_name
        logger.debug("next_agent_set", agent=agent_name)

    def set_next_worker(self, worker_name: str) -> None:
        """设置下一个 Worker

        Args:
            worker_name: 目标 Worker 名称
        """
        self._update.next_worker = worker_name
        logger.debug("next_worker_set", worker=worker_name)

    def set_handoff_target(self, target: str | None) -> None:
        """设置切换目标

        Args:
            target: 目标 Agent 名称，None 表示取消切换
        """
        self._update.handoff_target = target
        logger.debug("handoff_target_set", target=target)

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据

        Args:
            key: 键
            value: 值
        """
        self._update.metadata[key] = value

    def can_continue(self) -> bool:
        """检查是否可以继续执行

        运行所有验证器，全部通过才返回 True。

        Returns:
            是否可以继续
        """
        for validator in self._validators:
            if not validator(self._state):
                return False
        return True

    def should_terminate(self) -> bool:
        """检查是否应该终止

        Returns:
            是否应该终止
        """
        return not self.can_continue()

    def get_updates(self) -> dict[str, Any]:
        """获取状态更新字典

        Returns:
            LangGraph 兼容的状态更新字典
        """
        return self._update.to_dict()

    def has_updates(self) -> bool:
        """检查是否有待应用的更新

        Returns:
            是否有更新
        """
        return not self._update.is_empty()

    def reset_updates(self) -> None:
        """重置更新（不应用）"""
        self._update = StateUpdate()


# 便捷装饰器
def state_update(
    require_messages: bool = False,
    auto_increment: bool = False,
) -> Callable:
    """状态更新装饰器

    简化节点函数的状态更新逻辑。

    Args:
        require_messages: 是否要求必须有消息更新
        auto_increment: 是否自动递增迭代计数

    Returns:
        装饰后的函数

    Examples:
        ```python
        @state_update(auto_increment=True)
        async def my_node(state: AgentState, config: RunnableConfig) -> dict:
            # 函数逻辑
            return {"messages": [AIMessage(content="...")]}
        ```
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(state: AgentState, config, **kwargs) -> dict:
            manager = StateManager(state)

            # 调用原始函数
            result = await func(state, config, **kwargs)

            # 处理返回值
            if isinstance(result, dict):
                # 自动递增
                if auto_increment:
                    manager.increment_iteration()

                # 处理消息
                messages = result.get("messages", [])
                if messages:
                    if isinstance(messages, list):
                        manager.add_messages(messages)
                    else:
                        manager.add_message(messages)

                # 处理其他字段
                for key, value in result.items():
                    if key == "messages":
                        continue
                    if key == "_next_agent":
                        manager.set_next_agent(value)
                    elif key == "_next_worker":
                        manager.set_next_worker(value)
                    elif key == "_handoff_target":
                        manager.set_handoff_target(value)
                    else:
                        manager.set_metadata(key, value)

                # 验证
                if require_messages and not manager._update.messages:
                    logger.warning("no_messages_in_update", function=func.__name__)

                return manager.get_updates()

            return result

        return wrapper

    return decorator


def create_state_update(**kwargs) -> dict[str, Any]:
    """创建状态更新字典的便捷函数

    Args:
        **kwargs: 状态字段

    Returns:
        状态更新字典

    Examples:
        ```python
        update = create_state_update(
            messages=[AIMessage(content="...")],
            iteration_count=1,
            _next_agent="worker_a",
        )
        ```
    """
    update = StateUpdate()

    if "messages" in kwargs:
        messages = kwargs["messages"]
        if isinstance(messages, list):
            update.messages.extend(messages)
        else:
            update.messages.append(messages)

    if "iteration_count" in kwargs:
        update.iteration_delta = kwargs["iteration_count"]

    if "_next_agent" in kwargs:
        update.next_agent = kwargs["_next_agent"]

    if "_next_worker" in kwargs:
        update.next_worker = kwargs["_next_worker"]

    if "_handoff_target" in kwargs:
        update.handoff_target = kwargs["_handoff_target"]

    # 其他元数据
    for key, value in kwargs.items():
        if key not in (
            "messages",
            "iteration_count",
            "_next_agent",
            "_next_worker",
            "_handoff_target",
        ):
            update.metadata[key] = value

    return update.to_dict()


__all__ = [
    # 状态管理器
    "StateManager",
    # 状态更新
    "StateUpdate",
    # 验证器
    "StateValidator",
    # 便捷函数
    "state_update",
    "create_state_update",
]
