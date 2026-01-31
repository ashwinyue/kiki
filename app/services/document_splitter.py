"""文档分块器

基于 LangChain RecursiveCharacterTextSplitter 实现文档分块功能。
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    """分块配置

    Attributes:
        chunk_size: 块大小（字符数）
        chunk_overlap: 块重叠大小
        separators: 分隔符列表（按优先级排序）
        length_function: 长度计算函数
        keep_separator: 是否保留分隔符
        strip_whitespace: 是否去除空白
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = dataclass_field(
        default_factory=lambda: [
            "\n\n",  # 段落分隔
            "\n",  # 行分隔
            "。",  # 中文句号
            ". ",  # 英文句号
            "！",
            "! ",
            "？",
            "? ",
            "；",
            "; ",
            "，",
            ", ",
            " ",  # 空格
            "",  # 字符级分割
        ]
    )
    length_function: callable = len
    keep_separator: bool = False
    strip_whitespace: bool = True

    def validate(self) -> None:
        """验证配置

        Raises:
            ValueError: 配置无效
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        if not self.separators:
            raise ValueError("separators cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators,
            "keep_separator": self.keep_separator,
            "strip_whitespace": self.strip_whitespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkConfig":
        """从字典创建配置

        Args:
            data: 配置字典

        Returns:
            ChunkConfig 实例
        """
        return cls(
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 200),
            separators=data.get(
                "separators",
                ["\n\n", "\n", "。", ". ", " ", ""],
            ),
            keep_separator=data.get("keep_separator", False),
            strip_whitespace=data.get("strip_whitespace", True),
        )


@dataclass
class DocumentChunk:
    """文档分块

    Attributes:
        content: 分块内容
        metadata: 元数据
        chunk_index: 分块索引
        start_pos: 在原文档中的起始位置
        end_pos: 在原文档中的结束位置
    """

    content: str
    metadata: dict[str, Any]
    chunk_index: int
    start_pos: int = 0
    end_pos: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }


class DocumentSplitter:
    """文档分块器

    使用递归字符分割策略，保持语义完整性。
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """初始化分块器

        Args:
            config: 分块配置
        """
        self.config = config or ChunkConfig()
        self.config.validate()

        # 创建 LangChain 分割器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=self.config.length_function,
            keep_separator=self.config.keep_separator,
            strip_whitespace=self.config.strip_whitespace,
        )

        logger.info(
            "document_splitter_created",
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[DocumentChunk]:
        """分割文本

        Args:
            text: 待分割的文本
            metadata: 文档元数据

        Returns:
            DocumentChunk 列表
        """
        if not text:
            logger.warning("split_empty_text")
            return []

        # 使用 LangChain 分割
        split_docs = self._splitter.create_documents(
            [text],
            metadatas=[metadata or {}],
        )

        chunks = []
        current_pos = 0

        for i, doc in enumerate(split_docs):
            # 计算位置（近似）
            start_pos = current_pos
            end_pos = start_pos + len(doc.page_content)

            chunk = DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                chunk_index=i,
                start_pos=start_pos,
                end_pos=end_pos,
            )
            chunks.append(chunk)

            # 更新位置（考虑重叠）
            current_pos = end_pos - self.config.chunk_overlap

        logger.info(
            "text_split",
            original_length=len(text),
            chunk_count=len(chunks),
        )

        return chunks

    def split_documents(
        self,
        documents: list[Any],
    ) -> list[DocumentChunk]:
        """分割 LangChain 文档列表

        Args:
            documents: LangChain Document 对象列表

        Returns:
            DocumentChunk 列表
        """
        if not documents:
            return []

        # 使用 LangChain 分割
        split_docs = self._splitter.split_documents(documents)

        chunks = []
        for i, doc in enumerate(split_docs):
            chunk = DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                chunk_index=i,
            )
            chunks.append(chunk)

        logger.info(
            "documents_split",
            original_count=len(documents),
            chunk_count=len(chunks),
        )

        return chunks

    async def split_from_loader(
        self,
        loader: Any,  # BaseLoader from document_loaders
    ) -> list[DocumentChunk]:
        """从加载器加载并分割文档

        Args:
            loader: 文档加载器

        Returns:
            DocumentChunk 列表
        """
        result = await loader.load()
        # 从加载结果创建文档
        from langchain_core.documents import Document

        documents = [Document(page_content=result.content, metadata=result.metadata)]
        return self.split_documents(documents)

    def update_config(self, **kwargs: Any) -> None:
        """更新配置

        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.config.validate()

        # 重新创建分割器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=self.config.length_function,
            keep_separator=self.config.keep_separator,
            strip_whitespace=self.config.strip_whitespace,
        )

        logger.info(
            "splitter_config_updated",
            config=self.config.to_dict(),
        )


class CodeSplitter(DocumentSplitter):
    """代码文档分块器

    针对代码优化的分块器，保持代码结构完整性。
    """

    CODE_SEPARATORS = [
        "\n\n\n",  # 多空行（类/函数之间）
        "\n\n",  # 双空行
        "\n",  # 单空行
        "    ",  # 缩进
        " ",  # 空格
        "",  # 字符级
    ]

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        language: str = "python",
    ) -> None:
        """初始化代码分块器

        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠
            language: 编程语言（影响分隔符选择）
        """
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._get_separators_for_language(language),
        )

        super().__init__(config)

    def _get_separators_for_language(self, language: str) -> list[str]:
        """根据编程语言获取分隔符

        Args:
            language: 编程语言

        Returns:
            分隔符列表
        """
        # 基础分隔符
        base_separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            "    ",
            " ",
            "",
        ]

        # 语言特定分隔符
        lang_separators: dict[str, list[str]] = {
            "python": [
                "\n\nclass ",
                "\ndef ",
                "\n\n# ",
                "\n# ",
                "\n\n",
                "\n",
                "    ",
                " ",
                "",
            ],
            "javascript": [
                "\n\nfunction ",
                "\n\nconst ",
                "\n\nlet ",
                "\n\nvar ",
                "\n\n",
                "\n",
                "  ",
                " ",
                "",
            ],
            "java": [
                "\n\npublic class ",
                "\n\nprivate ",
                "\n\npublic ",
                "\n\nprotected ",
                "\n\n",
                "\n",
                "    ",
                " ",
                "",
            ],
            "go": [
                "\n\nfunc ",
                "\n\ntype ",
                "\n\nconst ",
                "\n\nvar ",
                "\n\n",
                "\n",
                "\t",
                " ",
                "",
            ],
        }

        return lang_separators.get(language.lower(), base_separators)


class MarkdownSplitter(DocumentSplitter):
    """Markdown 文档分块器

    针对 Markdown 优化的分块器，保持标题结构完整性。
    """

    MARKDOWN_SEPARATORS = [
        "\n## ",  # 二级标题
        "\n### ",  # 三级标题
        "\n#### ",  # 四级标题
        "\n##### ",  # 五级标题
        "\n###### ",  # 六级标题
        "\n# ",  # 一级标题
        "\n\n\n",  # 多空行
        "\n\n",  # 双空行
        "\n",  # 单空行
        "- ",  # 列表
        "* ",  # 列表
        "1. ",  # 有序列表
        " ",  # 空格
        "",  # 字符级
    ]

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> None:
        """初始化 Markdown 分块器

        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠
        """
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.MARKDOWN_SEPARATORS,
        )

        super().__init__(config)


# 预定义配置
PRESET_CONFIGS: dict[str, ChunkConfig] = {
    "default": ChunkConfig(),
    "small": ChunkConfig(chunk_size=500, chunk_overlap=50),
    "large": ChunkConfig(chunk_size=2000, chunk_overlap=400),
    "code": ChunkConfig(
        chunk_size=1500,
        chunk_overlap=300,
        separators=CodeSplitter.CODE_SEPARATORS,
    ),
    "markdown": ChunkConfig(
        chunk_size=1500,
        chunk_overlap=200,
        separators=MarkdownSplitter.MARKDOWN_SEPARATORS,
    ),
}


def get_preset_config(name: str) -> ChunkConfig:
    """获取预定义配置

    Args:
        name: 配置名称

    Returns:
        ChunkConfig 实例

    Raises:
        KeyError: 未找到配置
    """
    if name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    config = PRESET_CONFIGS[name]
    # 返回副本避免修改原配置
    return ChunkConfig(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators.copy(),
        keep_separator=config.keep_separator,
        strip_whitespace=config.strip_whitespace,
    )


async def split_text(
    text: str,
    config: ChunkConfig | None = None,
    preset: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """便捷函数：分割文本

    Args:
        text: 待分割的文本
        config: 分块配置
        preset: 预定义配置名称（优先级高于 config）
        metadata: 文档元数据

    Returns:
        DocumentChunk 列表

    Examples:
        ```python
        # 使用默认配置
        chunks = await split_text("long text...")

        # 使用预定义配置
        chunks = await split_text("long text...", preset="markdown")

        # 使用自定义配置
        config = ChunkConfig(chunk_size=500, chunk_overlap=100)
        chunks = await split_text("long text...", config=config)
        ```
    """
    if preset:
        config = get_preset_config(preset)

    splitter = DocumentSplitter(config)
    return splitter.split_text(text, metadata)


__all__ = [
    "ChunkConfig",
    "DocumentChunk",
    "DocumentSplitter",
    "CodeSplitter",
    "MarkdownSplitter",
    "PRESET_CONFIGS",
    "get_preset_config",
    "split_text",
]
