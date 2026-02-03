"""
高级图构建器 - 三层记忆架构工作流

构建支持三层记忆架构的 LangGraph 工作流：
1. 第一层整合：原文→摘要（局部精炼）
2. 第二层整合：摘要→结构规划（全局组织）
3. 第三层整合：基于规划重新生成（深度融合）

参考：aold/ai-engineer-training2/week01/code/05-2langgraph.py
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint

from app.agent.graph.advanced_nodes import NODES
from app.agent.state.advanced import AdvancedGenerationState
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AdvancedGenerationBuilder:
    """高级生成工作流构建器

    构建三层记忆架构的 LangGraph 工作流。
    """

    def __init__(
        self,
        checkpoint_saver: BaseCheckpointSaver | None = None,
    ):
        """初始化构建器

        Args:
            checkpoint_saver: 检查点保存器（可选）
        """
        self.checkpoint_saver = checkpoint_saver
        self._workflow: StateGraph | None = None

        logger.info(
            "advanced_generation_builder_initialized",
            has_checkpoint=checkpoint_saver is not None,
        )

    def build(self) -> StateGraph:
        """构建工作流图

        Returns:
            编译后的 StateGraph
        """
        workflow = StateGraph(AdvancedGenerationState)

        # 添加节点
        for node_name, node_func in NODES.items():
            workflow.add_node(node_name, node_func)
            logger.debug("node_added", name=node_name)

        # 设置入口点
        workflow.set_entry_point("split")

        # 添加边（线性流程）
        workflow.add_edge("split", "summarize_and_memorize")
        workflow.add_edge("summarize_and_memorize", "plan")
        workflow.add_edge("plan", "generate")
        workflow.add_edge("generate", END)

        self._workflow = workflow

        logger.info("workflow_built")

        return workflow

    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Any:
        """编译工作流

        Args:
            checkpointer: 检查点保存器（可选，覆盖初始化时的设置）

        Returns:
            编译后的可运行工作流
        """
        if self._workflow is None:
            self.build()

        checkpointer = checkpointer or self.checkpoint_saver

        compiled = self._workflow.compile(checkpointer=checkpointer)

        logger.info(
            "workflow_compiled",
            has_checkpointer=checkpointer is not None,
        )

        return compiled


def create_advanced_generation_workflow(
    checkpoint_saver: BaseCheckpointSaver | None = None,
) -> Any:
    """便捷函数：创建高级生成工作流

    Args:
        checkpoint_saver: 检查点保存器（可选）

    Returns:
        编译后的可运行工作流

    Example:
        ```python
        from app.agent.graph.advanced_builder import create_advanced_generation_workflow

        # 创建工作流
        workflow = create_advanced_generation_workflow()

        # 执行
        initial_state = {
            "original_text": "长文本内容...",
            "chunks": [],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        result = await workflow.ainvoke(initial_state)
        print(result["final_output"])
        ```
    """
    builder = AdvancedGenerationBuilder(checkpoint_saver=checkpoint_saver)
    return builder.compile()


async def run_advanced_generation(
    original_text: str,
    config: RunnableConfig | None = None,
    checkpoint_saver: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    """便捷函数：运行高级生成工作流

    Args:
        original_text: 原始输入文本
        config: RunnableConfig（可选）
        checkpoint_saver: 检查点保存器（可选）

    Returns:
        生成结果字典，包含 final_output 和 metadata

    Example:
        ```python
        from app.agent.graph.advanced_builder import run_advanced_generation

        result = await run_advanced_generation("长文本内容...")
        print(result["final_output"])
        print(f"段落数: {result['metadata']['section_count']}")
        ```
    """
    workflow = create_advanced_generation_workflow(checkpoint_saver=checkpoint_saver)

    initial_state: AdvancedGenerationState = {
        "original_text": original_text,
        "chunks": [],
        "summaries": [],
        "planning_tree": {},
        "final_output": "",
        "vectorstore": None,
        "metadata": {
            "chunk_count": 0,
            "summary_count": 0,
            "section_count": 0,
            "compression_ratio": 0.0,
        },
        "current_step": "",
        "error": None,
    }

    result = await workflow.ainvoke(initial_state, config=config)

    logger.info(
        "advanced_generation_completed",
        final_length=len(result.get("final_output", "")),
        metadata=result.get("metadata", {}),
    )

    return {
        "output": result.get("final_output", ""),
        "metadata": result.get("metadata", {}),
        "current_step": result.get("current_step", ""),
    }


# 导出
__all__ = [
    "AdvancedGenerationBuilder",
    "create_advanced_generation_workflow",
    "run_advanced_generation",
]
