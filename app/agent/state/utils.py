"""状态工具函数

提供状态处理的辅助函数。
"""

from typing import Any

# 导出 add_messages 以兼容 LangGraph
from langgraph.graph.message import add_messages


def should_stop_iteration(state: dict[str, Any]) -> bool:
    """检查是否应该停止迭代

    Args:
        state: 当前状态

    Returns:
        是否应该停止
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    return iteration_count >= max_iterations


def increment_iteration(state: dict[str, Any]) -> dict[str, Any]:
    """增加迭代计数（带验证）

    Args:
        state: 当前状态

    Returns:
        更新后的状态增量
    """
    current = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    if current >= max_iterations:
        from app.observability.logging import get_logger

        logger = get_logger(__name__)
        logger.warning(
            "max_iterations_reached",
            iteration_count=current,
            max_iterations=max_iterations,
        )
        return {}

    return {"iteration_count": current + 1}


def merge_lists(
    state: dict[str, Any],
    key: str,
    new_items: list[Any],
) -> dict[str, Any]:
    """合并列表（内部使用）

    Args:
        state: 当前状态
        key: 列表键
        new_items: 新项目

    Returns:
        更新后的状态增量
    """
    existing = state.get(key, [])
    return {key: existing + new_items}


__all__ = [
    "should_stop_iteration",
    "increment_iteration",
    "merge_lists",
    # 向后兼容导出
    "add_messages",
]
