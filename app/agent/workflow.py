# Copyright (c) 2025 Kiki. All rights reserved.

"""Agent 工作流入口

参考 DeerFlow 的 workflow.py 设计，提供工作流执行入口。

使用示例:
```python
from app.agent.workflow import run_agent_workflow

result = await run_agent_workflow(
    user_input="帮我查询天气",
    session_id="session-123",
    user_id="user-456",
)
```
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.types import RunnableConfig

from app.agent.graph import build_chat_graph, build_graph, build_graph_with_memory
from app.agent.graph.state import State, create_initial_state, create_state_from_input
from app.agent.graph.utils import needs_clarification
from app.config.settings import get_settings
from app.llm import get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# 创建全局图实例
graph = build_graph()


async def run_agent_workflow(
    user_input: str,
    session_id: str,
    user_id: str | None = None,
    debug: bool = False,
    enable_memory: bool = True,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """运行 Agent 工作流

    参考 DeerFlow 的 run_agent_workflow_async 设计。

    Args:
        user_input: 用户输入
        session_id: 会话 ID
        user_id: 用户 ID
        debug: 是否启用调试日志
        enable_memory: 是否启用持久化
        system_prompt: 系统提示词

    Returns:
        最终状态
    """
    if debug:
        logging.getLogger("app.agent").setLevel(logging.DEBUG)

    logger.info(
        "workflow_started",
        session_id=session_id,
        user_id=user_id,
    )

    # 选择图实例
    if enable_memory:
        workflow_graph = build_graph_with_memory()
    else:
        workflow_graph = build_graph()

    # 准备输入状态
    initial_state = create_state_from_input(
        input_text=user_input,
        user_id=user_id,
        session_id=session_id,
    )

    # 准备配置
    config = RunnableConfig(
        configurable={"thread_id": session_id},
        metadata={
            "user_id": user_id,
            "session_id": session_id,
        },
    )

    # 执行工作流
    final_state = None
    async for state in workflow_graph.astream(
        input=initial_state,
        config=config,
        stream_mode="values",
    ):
        final_state = state

    # 检查是否需要澄清
    if final_state and needs_clarification(final_state):
        logger.info(
            "clarification_needed",
            rounds=final_state.get("clarification_rounds", 0),
        )

    logger.info("workflow_completed", session_id=session_id)

    return final_state


async def run_agent_workflow_stream(
    user_input: str,
    session_id: str,
    user_id: str | None = None,
    debug: bool = False,
    enable_memory: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """运行 Agent 工作流（流式）

    Args:
        user_input: 用户输入
        session_id: 会话 ID
        user_id: 用户 ID
        debug: 是否启用调试日志
        enable_memory: 是否启用持久化

    Yields:
        状态更新
    """
    if debug:
        logging.getLogger("app.agent").setLevel(logging.DEBUG)

    logger.info(
        "workflow_stream_started",
        session_id=session_id,
        user_id=user_id,
    )

    # 选择图实例
    if enable_memory:
        workflow_graph = build_graph_with_memory()
    else:
        workflow_graph = build_graph()

    # 准备输入状态
    initial_state = create_state_from_input(
        input_text=user_input,
        user_id=user_id,
        session_id=session_id,
    )

    # 准备配置
    config = RunnableConfig(
        configurable={"thread_id": session_id},
        metadata={
            "user_id": user_id,
            "session_id": session_id,
        },
    )

    # 流式执行
    async for state in workflow_graph.astream(
        input=initial_state,
        config=config,
        stream_mode="values",
    ):
        yield state

    logger.info("workflow_stream_completed", session_id=session_id)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await run_agent_workflow(
            user_input="你好",
            session_id="test-session",
            debug=True,
        )
        print(f"Final state: {result}")

    asyncio.run(main())
