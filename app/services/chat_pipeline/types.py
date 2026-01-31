"""聊天管道类型定义

对齐 WeKnora99 ChatManage 类型定义。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchResultType(str, Enum):
    """搜索结果类型

    对齐 WeKnora99 MatchType
    """

    VECTOR = "vector"  # 向量搜索匹配
    KEYWORD = "keyword"  # 关键词搜索匹配
    HYBRID = "hybrid"  # 混合搜索匹配
    DIRECT_LOAD = "direct_load"  # 直接加载（小文件）
    HISTORY = "history"  # 历史对话匹配
    WEB_SEARCH = "web_search"  # 网络搜索
    FAQ = "faq"  # FAQ 匹配


class SearchTargetType(str, Enum):
    """搜索目标类型

    对齐 WeKnora99 SearchTargetType
    """

    KNOWLEDGE_BASE = "knowledge_base"  # 整个知识库
    KNOWLEDGE = "knowledge"  # 指定知识条目
    TAG = "tag"  # 按标签搜索


@dataclass
class SearchTarget:
    """搜索目标

    对齐 WeKnora99 SearchTarget
    """

    knowledge_base_id: str
    target_type: SearchTargetType
    knowledge_ids: list[str] = field(default_factory=list)
    tag_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "target_type": self.target_type.value,
            "knowledge_ids": self.knowledge_ids,
            "tag_ids": self.tag_ids,
        }


@dataclass
class SearchResult:
    """搜索结果

    对齐 WeKnora99 SearchResult
    """

    id: str  # Chunk ID
    content: str
    score: float
    knowledge_id: str | None = None
    knowledge_title: str | None = None
    knowledge_filename: str | None = None
    knowledge_source: str | None = None
    chunk_index: int | None = None
    match_type: SearchResultType = SearchResultType.HYBRID
    chunk_type: str | None = None
    parent_chunk_id: str | None = None
    image_info: str | None = None  # JSON string
    chunk_metadata: bytes | None = None  # JSON bytes
    start_at: int | None = None
    end_at: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "knowledge_id": self.knowledge_id,
            "knowledge_title": self.knowledge_title,
            "knowledge_filename": self.knowledge_filename,
            "knowledge_source": self.knowledge_source,
            "chunk_index": self.chunk_index,
            "match_type": self.match_type.value,
            "chunk_type": self.chunk_type,
            "parent_chunk_id": self.parent_chunk_id,
            "image_info": self.image_info,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "metadata": self.metadata,
        }


@dataclass
class History:
    """对话历史

    对齐 WeKnora99 History
    """

    query: str
    answer: str
    knowledge_references: list[SearchResult] = field(default_factory=list)
    created_at: str | None = None


class ChatPipelineConfig(BaseModel):
    """聊天管道配置

    对齐 WeKnora99 ChatManage 配置
    """

    # 基础配置
    tenant_id: int
    session_id: str

    # 模型配置
    chat_model_id: str | None = None
    rerank_model_id: str | None = None
    embedding_model_id: str | None = None

    # 搜索配置
    enable_search: bool = True
    enable_rewrite: bool = True
    enable_web_search: bool = False

    # 搜索目标
    knowledge_base_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    tag_ids: list[int] = Field(default_factory=list)

    # 搜索参数
    embedding_top_k: int = 20
    rerank_top_k: int = 5
    vector_threshold: float = 0.3
    keyword_threshold: float = 0.3
    rerank_threshold: float = 0.5

    # 查询扩展
    enable_query_expansion: bool = True

    # FAQ 优先级
    faq_priority_enabled: bool = False
    faq_score_boost: float = 1.2

    # 上下文配置
    max_context_length: int = 4000
    max_history_rounds: int = 5

    # 提示词
    system_prompt: str = "你是一个有用的助手。"
    rewrite_prompt_system: str | None = None
    rewrite_prompt_user: str | None = None

    # 流式响应
    stream: bool = False

    class Config:
        populate_by_name = True


@dataclass
class ChatContext:
    """聊天上下文

    对齐 WeKnora99 ChatManage 运行时状态
    """

    config: ChatPipelineConfig
    query: str
    rewrite_query: str = ""
    search_targets: list[SearchTarget] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    rerank_results: list[SearchResult] = field(default_factory=list)
    history: list[History] = field(default_factory=list)
    context_str: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "rewrite_query": self.rewrite_query,
            "search_results_count": len(self.search_results),
            "rerank_results_count": len(self.rerank_results),
            "history_rounds": len(self.history),
            "context_length": len(self.context_str),
            "error": self.error,
        }


@dataclass
class ChatPipelineResult:
    """聊天管道结果"""

    answer: str
    sources: list[SearchResult]
    context: ChatContext
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }


__all__ = [
    "SearchResultType",
    "SearchTargetType",
    "SearchTarget",
    "SearchResult",
    "History",
    "ChatPipelineConfig",
    "ChatContext",
    "ChatPipelineResult",
]
