"""意图澄清模块

当用户意图不明确时，通过多轮对话澄清用户需求。
参考 DeerFlow 的 Clarification 机制设计。

使用示例:
    ```python
    from app.agent.clarification import (
        needs_clarification,
        build_clarified_query,
        create_clarification_prompt,
        record_clarification,
    )

    # 检查是否需要澄清
    state = {"enable_clarification": True, "clarification_rounds": 1}
    if needs_clarification(state):
        # 构建澄清提示
        prompt = create_clarification_prompt(
            original_query="查询天气",
            clarification_history=["北京"],
        )

    # 记录澄清响应
    new_state = record_clarification(state, "朝阳区")

    # 构建完整查询
    full_query = build_clarified_query(
        original="查询天气",
        history=["北京", "朝阳区"]
    )
    # => "查询天气 - 北京, 朝阳区"
    ```
"""

import logging
from typing import Any

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ==================== 澄清判断 ====================


def needs_clarification(state: dict[str, Any]) -> bool:
    """检查是否需要进行澄清

    Args:
        state: Agent 状态字典

    Returns:
        是否需要澄清

    Examples:
        >>> state = {"enable_clarification": True, "clarification_rounds": 1}
        >>> needs_clarification(state)
        True
    """
    # 检查是否启用澄清
    if not state.get("enable_clarification", False):
        return False

    # 检查澄清是否已完成
    if state.get("is_clarification_complete", False):
        return False

    # 检查是否在澄清轮次范围内
    clarification_rounds = state.get("clarification_rounds", 0)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)

    return 0 < clarification_rounds <= max_clarification_rounds


def should_prompt_clarification(
    last_message: str,
    state: dict[str, Any],
) -> bool:
    """判断是否应该提示澄清

    基于 LLM 的响应内容判断是否需要向用户澄清。

    Args:
        last_message: 最后一条 AI 消息
        state: Agent 状态字典

    Returns:
        是否应该提示澄清

    Examples:
        >>> msg = "请问您想查询哪个城市的天气？"
        >>> state = {"enable_clarification": True, "clarification_rounds": 0}
        >>> should_prompt_clarification(msg, state)
        True
    """
    if not state.get("enable_clarification", False):
        return False

    # 检查是否在最大轮次范围内
    clarification_rounds = state.get("clarification_rounds", 0)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)

    if clarification_rounds >= max_clarification_rounds:
        return False

    # 检查消息中是否包含澄清相关的关键词
    clarification_keywords = [
        "请问",
        "请告诉我",
        "需要了解",
        "能否提供",
        "您是指",
        "具体是",
        "哪个",
        "哪种",
        "请确认",
        "please",  # 英文
        "could you",
        "which one",
        "can you provide",
        "specific",
    ]

    last_message_lower = last_message.lower()
    return any(keyword in last_message_lower for keyword in clarification_keywords)


# ==================== 澄清历史管理 ====================


def record_clarification(
    state: dict[str, Any],
    user_response: str,
) -> dict[str, Any]:
    """记录用户澄清响应

    Args:
        state: 当前状态
        user_response: 用户澄清响应

    Returns:
        更新后的状态

    Examples:
        >>> state = {
        ...     "clarification_rounds": 1,
        ...     "clarification_history": ["天气"],
        ... }
        >>> new_state = record_clarification(state, "北京")
        >>> new_state["clarification_rounds"]
        2
        >>> new_state["clarification_history"]
        ['天气', '北京']
    """
    clarification_history = state.get("clarification_history", [])[:]
    clarification_rounds = state.get("clarification_rounds", 0)

    # 记录用户响应
    clarification_history.append(user_response)

    # 更新状态
    new_state = {
        **state,
        "clarification_history": clarification_history,
        "clarification_rounds": clarification_rounds + 1,
    }

    logger.info(
        "clarification_recorded",
        round=new_state["clarification_rounds"],
        history_length=len(clarification_history),
    )

    return new_state


def complete_clarification(state: dict[str, Any]) -> dict[str, Any]:
    """标记澄清完成

    Args:
        state: 当前状态

    Returns:
        更新后的状态

    Examples:
        >>> state = {"is_clarification_complete": False}
        >>> new_state = complete_clarification(state)
        >>> new_state["is_clarification_complete"]
        True
    """
    logger.info("clarification_completed")
    return {**state, "is_clarification_complete": True}


def reset_clarification(state: dict[str, Any]) -> dict[str, Any]:
    """重置澄清状态

    Args:
        state: 当前状态

    Returns:
        更新后的状态

    Examples:
        >>> state = {
        ...     "clarification_rounds": 3,
        ...     "clarification_history": ["a", "b"],
        ...     "is_clarification_complete": True,
        ... }
        >>> new_state = reset_clarification(state)
        >>> new_state["clarification_rounds"]
        0
        >>> new_state["clarification_history"]
        []
    """
    logger.info("clarification_reset")
    return {
        **state,
        "clarification_rounds": 0,
        "clarification_history": [],
        "is_clarification_complete": False,
    }


def build_clarified_query(
    original: str,
    history: list[str],
) -> str:
    """从澄清历史构建完整查询

    Args:
        original: 原始查询
        history: 澄清历史

    Returns:
        完整查询字符串

    Examples:
        >>> build_clarified_query("天气", ["北京", "朝阳区"])
        '天气 - 北京, 朝阳区'

        >>> build_clarified_query("天气", [])
        '天气'
    """
    if not history:
        return original

    return f"{original} - {', '.join(history)}"


def build_clarified_topic_from_history(
    clarification_history: list[str],
) -> tuple[str, list[str]]:
    """从澄清历史构建澄清后的主题字符串

    Args:
        clarification_history: 澄清历史列表

    Returns:
        (澄清后的主题字符串, 历史序列)

    Examples:
        >>> build_clarified_topic_from_history(["天气", "北京", "今天"])
        ('天气 - 北京, 今天', ['天气', '北京', '今天'])

        >>> build_clarified_topic_from_history(["天气"])
        ('天气', ['天气'])
    """
    sequence = [item for item in clarification_history if item]

    if not sequence:
        return "", []

    if len(sequence) == 1:
        return sequence[0], sequence

    head, *tail = sequence
    clarified_string = f"{head} - {', '.join(tail)}"

    return clarified_string, sequence


def get_clarification_summary(state: dict[str, Any]) -> str:
    """获取澄清摘要

    Args:
        state: Agent 状态

    Returns:
        澄清摘要字符串

    Examples:
        >>> state = {
        ...     "clarification_rounds": 2,
        ...     "clarification_history": ["天气", "北京"],
        ... }
        >>> get_clarification_summary(state)
        '澄清轮次: 2, 历史: [天气, 北京]'
    """
    rounds = state.get("clarification_rounds", 0)
    history = state.get("clarification_history", [])

    return f"澄清轮次: {rounds}, 历史: {history}"


# ==================== 澄清提示生成 ====================


CLARIFICATION_PROMPTS = {
    "zh-CN": {
        "default": "请提供更多细节，以便我更好地帮助您。",
        "ask_location": "请问您想查询哪个位置？",
        "ask_time": "请问您想查询什么时间？",
        "ask_category": "请问您指的是哪一类？",
        "confirm": "请确认您的需求是：{query}",
    },
    "en-US": {
        "default": "Please provide more details so I can help you better.",
        "ask_location": "Which location would you like to query?",
        "ask_time": "What time would you like to query?",
        "ask_category": "Which category are you referring to?",
        "confirm": "Please confirm your request: {query}",
    },
}


def create_clarification_prompt(
    original_query: str,
    clarification_history: list[str] | None = None,
    locale: str = "zh-CN",
    prompt_type: str = "default",
) -> str:
    """创建澄清提示

    Args:
        original_query: 原始查询
        clarification_history: 澄清历史
        locale: 语言环境
        prompt_type: 提示类型

    Returns:
        澄清提示字符串

    Examples:
        >>> create_clarification_prompt(
        ...     "天气",
        ...     clarification_history=["北京"],
        ...     prompt_type="ask_time"
        ... )
        '请问您想查询什么时间？'

        >>> create_clarification_prompt(
        ...     "天气",
        ...     prompt_type="confirm"
        ... )
        '请确认您的需求是：天气'
    """
    prompts = CLARIFICATION_PROMPTS.get(locale, CLARIFICATION_PROMPTS["zh-CN"])
    template = prompts.get(prompt_type, prompts["default"])

    # 替换占位符
    if "{query}" in template:
        full_query = build_clarified_query(
            original_query,
            clarification_history or [],
        )
        return template.format(query=full_query)

    return template


def format_clarification_context(
    state: dict[str, Any],
    original_query: str,
) -> str:
    """格式化澄清上下文，用于传递给 LLM

    Args:
        state: Agent 状态
        original_query: 原始查询

    Returns:
        格式化的上下文字符串

    Examples:
        >>> state = {
        ...     "clarification_rounds": 2,
        ...     "clarification_history": ["北京", "今天"],
        ...     "locale": "zh-CN",
        ... }
        >>> format_clarification_context(state, "天气")
        '当前查询: 天气 - 北京, 今天\\n澄清轮次: 2'
    """
    history = state.get("clarification_history", [])
    rounds = state.get("clarification_rounds", 0)
    locale = state.get("locale", "zh-CN")

    full_query = build_clarified_query(original_query, history)

    if locale == "zh-CN":
        lines = [
            f"当前查询: {full_query}",
            f"澄清轮次: {rounds}",
        ]
    else:
        lines = [
            f"Current query: {full_query}",
            f"Clarification rounds: {rounds}",
        ]

    return "\\n".join(lines)


# ==================== 澄清节点 ====================


class ClarificationNode:
    """澄清节点

    用于 LangGraph 中的澄清处理节点。
    """

    @staticmethod
    def should_clarify(state: dict[str, Any]) -> bool:
        """判断是否应该进入澄清流程

        Args:
            state: Agent 状态

        Returns:
            是否应该澄清
        """
        return needs_clarification(state)

    @staticmethod
    def increment_round(state: dict[str, Any]) -> dict[str, Any]:
        """增加澄清轮次

        Args:
            state: Agent 状态

        Returns:
            更新后的状态
        """
        rounds = state.get("clarification_rounds", 0)
        new_state = {**state, "clarification_rounds": rounds + 1}
        logger.info("clarification_round_incremented", new_rounds=rounds + 1)
        return new_state

    @staticmethod
    def get_next_action(state: dict[str, Any]) -> str:
        """获取下一个动作

        根据澄清状态决定下一步操作。

        Args:
            state: Agent 状态

        Returns:
            下一个动作名称

        Examples:
            >>> state = {
            ...     "clarification_rounds": 3,
            ...     "max_clarification_rounds": 3,
            ... }
            >>> ClarificationNode.get_next_action(state)
            'continue'

            >>> state = {"clarification_rounds": 1}
            >>> ClarificationNode.get_next_action(state)
            'clarify'
        """
        rounds = state.get("clarification_rounds", 0)
        max_rounds = state.get("max_clarification_rounds", 3)

        if rounds >= max_rounds:
            return "continue"  # 达到最大轮次，继续处理
        if state.get("is_clarification_complete", False):
            return "continue"  # 澄清完成，继续处理

        return "clarify"  # 继续澄清


# ==================== 导出 ====================


__all__ = [
    # 澄清判断
    "needs_clarification",
    "should_prompt_clarification",
    # 澄清历史管理
    "record_clarification",
    "complete_clarification",
    "reset_clarification",
    "build_clarified_query",
    "build_clarified_topic_from_history",
    "get_clarification_summary",
    # 澄清提示生成
    "create_clarification_prompt",
    "format_clarification_context",
    "CLARIFICATION_PROMPTS",
    # 澄清节点
    "ClarificationNode",
]
