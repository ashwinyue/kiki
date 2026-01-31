"""文档分块 Schema

对齐 WeKnora99 文档分块 API 规范
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============== 分块类型定义 ==============

ChunkType = str


class ChunkTypes:
    """分块类型常量"""

    TEXT = "text"
    IMAGE_OCR = "image_ocr"
    IMAGE_CAPTION = "image_caption"
    SUMMARY = "summary"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    FAQ = "faq"
    WEB_SEARCH = "web_search"
    TABLE_SUMMARY = "table_summary"
    TABLE_COLUMN = "table_column"


# ============== 请求 Schema ==============


class ChunkUpdate(BaseModel):
    """分块更新请求"""

    content: str | None = Field(None, description="分块内容")
    chunk_index: int | None = Field(None, alias="chunkIndex", description="分块索引")
    is_enabled: bool | None = Field(None, alias="isEnabled", description="是否启用")
    start_at: int | None = Field(None, alias="startAt", description="起始位置")
    end_at: int | None = Field(None, alias="endAt", description="结束位置")
    image_info: str | None = Field(None, alias="imageInfo", description="图片信息")
    metadata: Any | None = Field(None, description="元数据")


class DeleteQuestionRequest(BaseModel):
    """删除生成问题请求"""

    question_id: str = Field(..., alias="question_id", description="问题 ID")


# ============== 响应 Schema ==============


class ImageInfo(BaseModel):
    """图片信息"""

    url: str = ""
    original_url: str = Field("", alias="originalUrl")
    start_pos: int = Field(0, alias="startPos")
    end_pos: int = Field(0, alias="endPos")
    caption: str = ""
    ocr_text: str = Field("", alias="ocrText")


class ChunkResponse(BaseModel):
    """分块响应"""

    id: str
    seq_id: int | None = Field(None, alias="seqId")
    tenant_id: int
    knowledge_base_id: str = Field(alias="knowledgeBaseId")
    knowledge_id: str = Field(alias="knowledgeId")
    tag_id: str | None = Field(None, alias="tagId")

    content: str
    chunk_index: int = Field(alias="chunkIndex")
    is_enabled: bool = Field(alias="isEnabled")
    flags: int = 1
    status: int = 0
    start_at: int = Field(alias="startAt")
    end_at: int = Field(alias="endAt")

    pre_chunk_id: str | None = Field(None, alias="preChunkId")
    next_chunk_id: str | None = Field(None, alias="nextChunkId")
    chunk_type: ChunkType = Field(default=ChunkTypes.TEXT, alias="chunkType")
    parent_chunk_id: str | None = Field(None, alias="parentChunkId")

    relation_chunks: Any | None = Field(None, alias="relationChunks")
    indirect_relation_chunks: Any | None = Field(None, alias="indirectRelationChunks")
    metadata: Any | None = None
    content_hash: str | None = Field(None, alias="contentHash")
    image_info: str | None = Field(None, alias="imageInfo")

    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True


class ChunkListResponse(BaseModel):
    """分块列表响应"""

    items: list[ChunkResponse]
    total: int
    page: int
    page_size: int = Field(alias="pageSize")

    class Config:
        populate_by_name = True


__all__ = [
    "ChunkTypes",
    "ChunkType",
    "ChunkUpdate",
    "DeleteQuestionRequest",
    "ImageInfo",
    "ChunkResponse",
    "ChunkListResponse",
]
