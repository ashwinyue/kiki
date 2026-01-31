"""文档处理相关 Schema

定义文档解析请求和响应的数据结构。
"""

from typing import Any

from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    """文档分块配置"""

    preset: str | None = Field(
        None,
        description="预定义配置名称 (default, small, large, code, markdown)",
    )
    chunk_size: int = Field(1000, ge=100, le=10000, description="块大小")
    chunk_overlap: int = Field(200, ge=0, le=2000, description="块重叠大小")
    separators: list[str] | None = Field(
        None,
        description="自定义分隔符列表（按优先级排序）",
    )


class DocumentParseRequest(BaseModel):
    """文档解析请求"""

    # 文件来源（三选一）
    file_path: str | None = Field(None, description="本地文件路径")
    file_content: bytes | None = Field(None, description="文件内容（二进制）")
    filename: str | None = Field(None, description="文件名（与 file_content 一起使用）")

    # 可选参数
    format: str | None = Field(
        None,
        description="文档格式 (pdf, word, excel, ppt, text, markdown, web, image)",
    )
    chunk_config: ChunkConfig | None = Field(None, description="分块配置")


class DocumentUrlParseRequest(BaseModel):
    """网页内容解析请求"""

    url: str = Field(..., min_length=1, description="URL 地址")
    chunk_config: ChunkConfig | None = Field(None, description="分块配置")


class DocumentChunk(BaseModel):
    """文档分块"""

    content: str = Field(..., description="分块内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    chunk_index: int = Field(..., ge=0, description="分块索引")
    start_pos: int = Field(0, ge=0, description="在原文档中的起始位置")
    end_pos: int = Field(0, ge=0, description="在原文档中的结束位置")


class DocumentParseResponse(BaseModel):
    """文档解析响应"""

    success: bool = Field(..., description="是否成功")
    content: str = Field(..., description="完整文档内容")
    chunks: list[DocumentChunk] = Field(default_factory=list, description="文档分块列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    format: str = Field(..., description="文档格式")
    page_count: int = Field(0, ge=0, description="页数")
    source: str = Field(..., description="来源文件路径或 URL")
    chunk_count: int = Field(0, ge=0, description="分块数量")
    message: str | None = Field(None, description="处理消息")


class SupportedFormatInfo(BaseModel):
    """支持的格式信息"""

    name: str = Field(..., description="格式名称")
    extensions: list[str] = Field(..., description="文件扩展名")
    description: str = Field(..., description="格式描述")


class SupportedFormatsResponse(BaseModel):
    """支持格式列表响应"""

    formats: list[str] = Field(..., description="支持的格式列表")
    extensions: list[str] = Field(..., description="支持的文件扩展名")
    format_details: list[SupportedFormatInfo] = Field(
        default_factory=list,
        description="格式详细信息",
    )


class BatchParseRequest(BaseModel):
    """批量文档解析请求"""

    files: list[DocumentParseRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="待解析的文件列表（最多 10 个）",
    )
    chunk_config: ChunkConfig | None = Field(None, description="统一分块配置")


class BatchParseResponse(BaseModel):
    """批量解析响应"""

    total: int = Field(..., description="总文件数")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    results: list[DocumentParseResponse] = Field(
        ...,
        description="每个文件的解析结果",
    )


# 预定义格式信息
FORMAT_DETAILS = [
    SupportedFormatInfo(
        name="pdf",
        extensions=[".pdf"],
        description="PDF 文档",
    ),
    SupportedFormatInfo(
        name="word",
        extensions=[".doc", ".docx"],
        description="Word 文档",
    ),
    SupportedFormatInfo(
        name="excel",
        extensions=[".xls", ".xlsx", ".xlsm"],
        description="Excel 表格",
    ),
    SupportedFormatInfo(
        name="ppt",
        extensions=[".ppt", ".pptx"],
        description="PowerPoint 演示文稿",
    ),
    SupportedFormatInfo(
        name="text",
        extensions=[".txt"],
        description="纯文本",
    ),
    SupportedFormatInfo(
        name="markdown",
        extensions=[".md", ".markdown"],
        description="Markdown 文档",
    ),
    SupportedFormatInfo(
        name="image",
        extensions=[".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"],
        description="图片（OCR）",
    ),
    SupportedFormatInfo(
        name="web",
        extensions=[],
        description="网页内容",
    ),
]


__all__ = [
    "ChunkConfig",
    "DocumentParseRequest",
    "DocumentUrlParseRequest",
    "DocumentChunk",
    "DocumentParseResponse",
    "SupportedFormatInfo",
    "SupportedFormatsResponse",
    "BatchParseRequest",
    "BatchParseResponse",
    "FORMAT_DETAILS",
]
