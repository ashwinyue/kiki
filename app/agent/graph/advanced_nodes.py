"""
高级节点 - 三层记忆架构

实现三层记忆架构的核心节点：
1. 分块节点（split_node）
2. 摘要与记忆节点（summarize_and_memorize_node）
3. 规划节点（planning_node）
4. 生成节点（generate_node）

参考：aold/ai-engineer-training2/week01/code/05-2langgraph.py
"""

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from app.agent.state.advanced import (
    AdvancedGenerationState,
    DocumentSection,
    PlanningTree,
)
from app.agent.vector_store import (
    VectorStoreManager,
    retrieve_relevant_context,
)
from app.llm.service import LLMService
from app.observability.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Pydantic 模型用于结构化输出
# =============================================================================
class PlanningOutput(BaseModel):
    """规划输出模型"""

    title: str = Field(description="文档主标题")
    sections: list[dict[str, Any]] = Field(description="章节列表")
    total_paragraphs: int = Field(description="预计总段落数")


# =============================================================================
# 工具函数
# =============================================================================
def split_text_semantic(text: str, target_chunks: int = 8) -> list[str]:
    """语义化文本分块

    基于段落结构和语义完整性进行切分，确保切分范围在 2-10 块之间。

    Args:
        text: 输入文本
        target_chunks: 目标块数

    Returns:
        切分后的文本块列表
    """
    # 按段落分割，保留非空段落
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 如果段落数已在目标范围内，直接返回
    if 2 <= len(paragraphs) <= 10:
        return paragraphs

    # 如果段落太少，按句子进一步分割
    if len(paragraphs) < 2:
        sentences = []
        for para in paragraphs:
            # 按句号、问号、感叹号分割句子
            sent_list = re.split(r'[。！？]', para)
            sentences.extend([s.strip() for s in sent_list if s.strip()])

        # 将句子重新组合成 2-4 个块
        if len(sentences) >= 4:
            chunk_size = len(sentences) // 3
            chunks = []
            for i in range(0, len(sentences), chunk_size):
                chunk = "。".join(sentences[i:i + chunk_size])
                if chunk:
                    chunks.append(chunk + "。")
            return chunks[:10]  # 最多 10 块
        else:
            return sentences

    # 如果段落太多，合并相邻段落
    if len(paragraphs) > 10:
        chunk_size = len(paragraphs) // target_chunks
        chunks = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[i:i + chunk_size]
            chunks.append("\n\n".join(chunk_paras))
        return chunks

    return paragraphs


async def generate_summary_async(
    llm_service: LLMService,
    chunk: str,
    target_ratio: float = 0.3,
) -> str:
    """异步生成精简摘要

    Args:
        llm_service: LLM 服务
        chunk: 文本块
        target_ratio: 目标压缩比率（默认 30%）

    Returns:
        摘要文本
    """
    chunk_length = len(chunk)
    target_length = int(chunk_length * target_ratio)

    messages = [
        SystemMessage(content=f"""请对以下内容进行高度精简的摘要。要求：
1. 摘要长度不超过 {target_length} 字符（约原文的 {int(target_ratio * 100)}%）
2. 只保留最核心的观点和关键信息
3. 使用简洁的语言，避免冗余表达
4. 保持逻辑清晰，突出重点"""),
        HumanMessage(content=chunk),
    ]

    try:
        response = await llm_service.call(messages)
        summary = str(response.content)

        # 如果摘要仍然过长，进行二次压缩
        if len(summary) > target_length:
            messages = [
                SystemMessage(content=f"请将以下摘要进一步压缩到 {target_length} 字符以内，只保留最关键的信息："),
                HumanMessage(content=summary),
            ]
            response = await llm_service.call(messages)
            summary = str(response.content)

        return summary

    except Exception as e:
        logger.error("summary_generation_failed", error=str(e))
        # 回退：返回截断的原文
        return chunk[:target_length] + "..."


async def build_planning_tree_async(
    llm_service: LLMService,
    summaries: list[str],
) -> PlanningTree:
    """异步构建文章结构树

    Args:
        llm_service: LLM 服务
        summaries: 摘要列表

    Returns:
        文章结构树
    """
    combined = "\n\n".join(f"块 {i + 1}: {s}" for i, s in enumerate(summaries))

    prompt = f"""请根据以下文本块摘要，生成一份精简的综合报告结构大纲。

目的：
- 分析摘要内容，生成逻辑清晰的文章结构

要求：
- 总共只生成 3-4 个主要章节
- 每章不超过 1 个合并段落
- 将相关小节内容合并为综合性段落
- 保持逻辑连贯，突出核心内容
- 输出为严格 JSON 格式

摘要汇总：
{combined}

请只输出 JSON，格式如下（注意：subsections 为空数组）：
{{
  "title": "报告主标题",
  "sections": [
    {{"title": "发展现状与技术基础", "subsections": []}},
    {{"title": "应用领域与实践案例", "subsections": []}},
    {{"title": "挑战问题与未来趋势", "subsections": []}}
  ],
  "total_paragraphs": 3
}}"""

    messages = [HumanMessage(content=prompt)]

    try:
        # 使用结构化输出
        structured_llm = llm_service.with_structured_output(PlanningOutput)
        result: PlanningOutput = await structured_llm.ainvoke(messages)

        return {
            "title": result.title,
            "sections": [
                {
                    "title": s.get("title", ""),
                    "content": "",
                    "subsections": s.get("subsections", []),
                }
                for s in result.sections
            ],
            "total_paragraphs": result.total_paragraphs,
        }

    except Exception as e:
        logger.error("planning_tree_generation_failed", error=str(e))

        # 回退到默认结构
        return {
            "title": "文档分析报告",
            "sections": [
                {"title": "核心技术与发展现状", "content": "", "subsections": []},
                {"title": "应用实践与行业影响", "content": "", "subsections": []},
                {"title": "挑战机遇与未来展望", "content": "", "subsections": []},
            ],
            "total_paragraphs": 3,
        }


async def generate_section_content_async(
    llm_service: LLMService,
    section_title: str,
    context: str,
) -> str:
    """异步生成章节内容

    Args:
        llm_service: LLM 服务
        section_title: 章节标题
        context: 相关上下文

    Returns:
        章节内容
    """
    prompt = f"""你是专业撰稿人。请根据参考上下文，撰写以下章节的综合性内容。

# 上下文参考：
{context}

# 目标章节：
{section_title}

要求：
1. 将相关内容合并为一个完整的综合段落
2. 涵盖该主题的核心要点和关键信息
3. 语言精炼，逻辑清晰，避免冗余
4. 段落长度适中（200-400 字），内容丰富
5. 体现专业深度和分析价值"""

    messages = [HumanMessage(content=prompt)]

    try:
        response = await llm_service.call(messages)
        return str(response.content)

    except Exception as e:
        logger.error("section_content_generation_failed", section=section_title, error=str(e))
        return f"[{section_title}] 内容生成失败"


# =============================================================================
# LangGraph 节点定义
# =============================================================================
async def split_node(state: AdvancedGenerationState) -> AdvancedGenerationState:
    """分块节点：将原始文本切分为语义块

    这是三层记忆架构的第一步。
    """
    logger.info("split_node_started")

    chunks = split_text_semantic(state["original_text"])

    # 更新状态
    state["chunks"] = chunks
    state["current_step"] = "split"

    # 更新元数据
    state["metadata"]["chunk_count"] = len(chunks)
    avg_length = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
    state["metadata"]["avg_chunk_length"] = avg_length

    logger.info(
        "split_node_completed",
        chunk_count=len(chunks),
        avg_length=avg_length,
    )

    return state


async def summarize_and_memorize_node(
    state: AdvancedGenerationState,
    llm_service: LLMService | None = None,
) -> AdvancedGenerationState:
    """摘要与记忆节点：生成摘要并构建向量存储

    这是三层记忆架构的第一层整合：原文→摘要（局部精炼）
    """
    logger.info("summarize_and_memorize_node_started")

    if llm_service is None:
        from app.llm.service import get_llm_service
        llm_service = get_llm_service()

    summaries = []

    # 为每个文本块生成摘要
    for i, chunk in enumerate(state["chunks"]):
        summary = await generate_summary_async(llm_service, chunk)
        summaries.append(summary)

        compression_ratio = len(summary) / len(chunk) * 100 if chunk else 0
        logger.debug(
            "summary_generated",
            index=i + 1,
            original_length=len(chunk),
            summary_length=len(summary),
            compression_ratio=f"{compression_ratio:.1f}%",
        )

    state["summaries"] = summaries

    # 构建向量存储
    try:
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.create_from_texts(summaries)
        state["vectorstore"] = vector_store

        logger.info(
            "vector_store_created",
            doc_count=len(summaries),
        )

    except Exception as e:
        logger.error("vector_store_creation_failed", error=str(e))
        state["vectorstore"] = None

    # 更新元数据
    total_original = sum(len(c) for c in state["chunks"])
    total_summary = sum(len(s) for s in summaries)
    state["metadata"]["summary_count"] = len(summaries)
    state["metadata"]["compression_ratio"] = total_summary / total_original if total_original > 0 else 0

    state["current_step"] = "summarize_and_memorize"

    logger.info("summarize_and_memorize_node_completed")

    return state


async def planning_node(
    state: AdvancedGenerationState,
    llm_service: LLMService | None = None,
) -> AdvancedGenerationState:
    """规划节点：构建文章结构树

    这是三层记忆架构的第二层整合：摘要→结构规划（全局组织）
    """
    logger.info("planning_node_started")

    if llm_service is None:
        from app.llm.service import get_llm_service
        llm_service = get_llm_service()

    # 构建结构树
    planning_tree = await build_planning_tree_async(
        llm_service,
        state["summaries"],
    )

    state["planning_tree"] = planning_tree
    state["current_step"] = "planning"

    # 更新元数据
    state["metadata"]["section_count"] = len(planning_tree["sections"])

    logger.info(
        "planning_node_completed",
        title=planning_tree["title"],
        section_count=len(planning_tree["sections"]),
    )

    return state


async def generate_node(
    state: AdvancedGenerationState,
    llm_service: LLMService | None = None,
) -> AdvancedGenerationState:
    """生成节点：基于结构树生成最终内容

    这是三层记忆架构的第三层整合：基于规划重新生成（深度融合）
    """
    logger.info("generate_node_started")

    if llm_service is None:
        from app.llm.service import get_llm_service
        llm_service = get_llm_service()

    tree = state["planning_tree"]
    content_parts = []

    # 添加主标题
    if tree.get("title"):
        title = tree["title"]
        content_parts.append(f"# {title}\n")
        logger.debug("title_added", title=title)

    # 为每个章节生成内容
    for i, section in enumerate(tree["sections"], 1):
        sec_title = section.get("title", f"第 {i} 章")

        logger.debug("generating_section", index=i, title=sec_title)
        content_parts.append(f"## {sec_title}")

        # 从向量存储检索相关上下文
        context = retrieve_relevant_context(
            query=sec_title,
            vectorstore=state["vectorstore"],
            k=3,
        )

        # 生成章节内容
        content = await generate_section_content_async(
            llm_service,
            sec_title,
            context,
        )
        content_parts.append(content)

        # 更新章节内容
        section["content"] = content

        logger.debug(
            "section_generated",
            index=i,
            content_length=len(content),
        )

        # 将生成的内容添加到记忆库
        if state["vectorstore"] is not None:
            try:
                vector_manager = VectorStoreManager()
                vector_manager._store = state["vectorstore"]
                vector_manager.add_texts([content])
            except Exception as e:
                logger.warning("add_to_memory_failed", error=str(e))

    state["final_output"] = "\n\n".join(content_parts)
    state["current_step"] = "generate"

    # 更新元数据
    state["metadata"]["final_length"] = len(state["final_output"])

    logger.info(
        "generate_node_completed",
        final_length=len(state["final_output"]),
    )

    return state


# =============================================================================
# 导出的节点映射
# -*-
NODES = {
    "split": split_node,
    "summarize_and_memorize": summarize_and_memorize_node,
    "plan": planning_node,
    "generate": generate_node,
}

__all__ = [
    "split_node",
    "summarize_and_memorize_node",
    "planning_node",
    "generate_node",
    "NODES",
]
