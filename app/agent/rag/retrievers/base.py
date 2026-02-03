"""RAG 检索器基类

定义统一的检索器接口和结果数据模型。
参考 DeerFlow 的 Resource 设计，提供标准化的检索结果格式。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedDocument:
    """检索到的文档

    Attributes:
        title: 文档标题
        content: 文档内容
        source: 文档来源（URL、文件路径等）
        score: 相似度分数（可选）
        metadata: 额外元数据
    """
    title: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }

    def format_for_llm(self) -> str:
        """格式化为 LLM 可用的文本"""
        return f"[{self.title}]({self.source})\n{self.content}"


@dataclass
class RetrievalOptions:
    """检索选项

    Attributes:
        top_k: 返回结果数量
        score_threshold: 相似度分数阈值（低于此值的结果将被过滤）
        filter_kwargs: 额外过滤参数
    """
    top_k: int = 5
    score_threshold: float | None = None
    filter_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证参数"""
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if self.score_threshold is not None and not (0 <= self.score_threshold <= 1):
            raise ValueError("score_threshold 必须在 0-1 之间")


class BaseRetriever(ABC):
    """RAG 检索器基类

    所有检索器必须实现此接口，确保统一的调用方式。
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedDocument]:
        """检索相关文档

        Args:
            query: 查询文本
            options: 检索选项（可选）

        Returns:
            检索到的文档列表，按相关性降序排列

        Raises:
            RetrievalError: 检索失败时抛出
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查

        Returns:
            服务是否可用
        """
        pass

    async def ahealth_check(self) -> bool:
        """异步健康检查（默认使用同步方法）"""
        return self.health_check()

    def retrieve_sync(
        self,
        query: str,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedDocument]:
        """同步检索（默认实现，由子类覆盖以优化性能）

        Args:
            query: 查询文本
            options: 检索选项（可选）

        Returns:
            检索到的文档列表
        """
        import asyncio

        return asyncio.run(self.retrieve(query, options))


class RetrievalError(Exception):
    """检索错误

    当检索操作失败时抛出，包含详细的错误信息。
    """

    def __init__(self, message: str, retriever_type: str, cause: Exception | None = None):
        self.message = message
        self.retriever_type = retriever_type
        self.cause = cause
        super().__init__(f"[{retriever_type}] {message}")


__all__ = [
    "RetrievedDocument",
    "RetrievalOptions",
    "BaseRetriever",
    "RetrievalError",
]
