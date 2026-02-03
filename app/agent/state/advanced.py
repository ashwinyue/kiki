"""
增强的状态模型 - 支持三层记忆架构

三层记忆架构：
1. 第一层整合：原文→摘要（局部精炼）
2. 第二层整合：摘要→结构规划（全局组织）
3. 第三层整合：基于规划重新生成（深度融合）

参考：aold/ai-engineer-training2/week01/code/05-2langgraph.py
"""

from typing import Any, Required
from typing_extensions import TypedDict

from langchain_core.vectorstores import VectorStore


class AdvancedGenerationState(TypedDict):
    """增强的生成状态 - 支持多层记忆

    Attributes:
        original_text: 原始输入文本
        chunks: 切分后的文本块列表
        summaries: 每个文本块的摘要列表
        planning_tree: 生成的文章结构树（包含标题和章节）
        final_output: 最终生成的输出
        vectorstore: 向量数据库存储（用于语义检索）
        metadata: 附加元数据（如统计信息、配置参数等）
        current_step: 当前执行步骤（用于追踪进度）
        error: 错误信息（如果有）
    """

    # 输入
    original_text: Required[str]

    # 第一层：文本分块与摘要
    chunks: list[str]
    summaries: list[str]

    # 第二层：结构规划
    planning_tree: dict[str, Any]

    # 第三层：最终生成
    final_output: str

    # 向量存储（语义记忆）
    vectorstore: VectorStore | None

    # 元数据
    metadata: dict[str, Any]

    # 执行追踪
    current_step: str
    error: str | None


class DocumentSection(TypedDict):
    """文档章节结构

    Attributes:
        title: 章节标题
        content: 章节内容
        subsections: 子章节列表
    """

    title: str
    content: str
    subsections: list["DocumentSection"]


class PlanningTree(TypedDict):
    """文章结构树

    Attributes:
        title: 文档主标题
        sections: 主要章节列表
        total_paragraphs: 预计总段落数
    """

    title: str
    sections: list[DocumentSection]
    total_paragraphs: int


class MemoryContext(TypedDict):
    """记忆上下文

    Attributes:
        short_term: 短期记忆（当前对话上下文）
        long_term: 长期记忆（向量检索结果）
        working: 工作记忆（当前任务相关）
    """

    short_term: list[str]
    long_term: list[str]
    working: dict[str, Any]


class GenerationMetadata(TypedDict):
    """生成元数据

    Attributes:
        chunk_count: 分块数量
        summary_count: 摘要数量
        section_count: 章节数量
        total_tokens: 总 token 数
        compression_ratio: 压缩比率
        execution_time: 执行时间（秒）
    """

    chunk_count: int
    summary_count: int
    section_count: int
    total_tokens: int
    compression_ratio: float
    execution_time: float
