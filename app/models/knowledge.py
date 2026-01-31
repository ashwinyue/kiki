"""知识库相关模型

对齐 WeKnora99 表结构
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


# ============== Model (模型) ==============


class ModelBase(SQLModel):
    """模型基础模型"""

    name: str = Field(max_length=255)
    type: str = Field(max_length=50)  # Embedding, Rerank, KnowledgeQA, VLLM, Chat
    source: str = Field(max_length=50)  # local, remote, aliyun, zhipu, openai
    description: str | None = None
    is_default: bool = Field(default=False)
    is_builtin: bool = Field(default=False)
    status: str = Field(default="active", max_length=50)


class Model(ModelBase, table=True):
    """模型表模型"""

    __tablename__ = "models"

    id: str = Field(default=None, primary_key=True, max_length=64)
    tenant_id: int
    parameters: Any = Field(default={}, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class ModelCreate(ModelBase):
    """模型创建模型"""

    tenant_id: int
    parameters: Any = {}


# ============== KnowledgeBase (知识库) ==============


class KnowledgeBaseBase(SQLModel):
    """知识库基础模型"""

    name: str = Field(max_length=255)
    description: str | None = None
    kb_type: str = Field(default="document", max_length=32)  # document, faq


class KnowledgeBase(KnowledgeBaseBase, table=True):
    """知识库表模型"""

    __tablename__ = "knowledge_bases"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int

    # 配置
    chunking_config: Any = Field(
        default={"chunk_size": 512, "chunk_overlap": 50},
        sa_column=Column(JSONB),
    )
    image_processing_config: Any = Field(
        default={"enable_multimodal": False, "model_id": ""},
        sa_column=Column(JSONB),
    )

    # 关联模型
    embedding_model_id: str = Field(max_length=64)
    summary_model_id: str = Field(max_length=64)
    rerank_model_id: str | None = Field(default=None, max_length=64)

    # 其他配置
    cos_config: Any = Field(default={}, sa_column=Column(JSONB))
    vlm_config: Any = Field(default={}, sa_column=Column(JSONB))
    extract_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    faq_config: Any | None = Field(default=None, sa_column=Column(JSONB))
    question_generation_config: Any | None = Field(default=None, sa_column=Column(JSONB))

    is_temporary: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """知识库创建模型"""

    tenant_id: int
    embedding_model_id: str
    summary_model_id: str
    chunking_config: Any | None = None


# ============== Knowledge (知识条目) ==============


class KnowledgeBase(SQLModel):
    """知识条目基础模型"""

    type: str = Field(max_length=50)  # file, url, text, faq
    title: str = Field(max_length=255)
    description: str | None = None
    source: str = Field(max_length=128)


class Knowledge(KnowledgeBase, table=True):
    """知识条目表模型"""

    __tablename__ = "knowledges"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    knowledge_base_id: str = Field(max_length=36)

    parse_status: str = Field(default="unprocessed", max_length=50)
    enable_status: str = Field(default="enabled", max_length=50)

    embedding_model_id: str | None = Field(default=None, max_length=64)
    tag_id: str | None = Field(default=None, max_length=36)
    summary_status: str = Field(default="none", max_length=32)

    # 文件信息
    file_name: str | None = Field(default=None, max_length=255)
    file_type: str | None = Field(default=None, max_length=50)
    file_size: int | None = None
    file_path: str | None = None
    file_hash: str | None = Field(default=None, max_length=64)
    storage_size: int = Field(default=0)

    meta_data: Any | None = Field(default=None, sa_column=Column(JSONB))
    last_faq_import_result: Any | None = Field(default=None, sa_column=Column(JSONB))

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = Field(default=None)
    error_message: str | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)


class KnowledgeCreate(SQLModel):
    """知识条目创建模型"""

    tenant_id: int
    knowledge_base_id: str
    type: str
    title: str
    source: str
    file_name: str | None = None
    file_type: str | None = None


# ============== Chunk (文档分块) ==============


class ChunkBase(SQLModel):
    """文档分块基础模型"""

    content: str
    chunk_index: int
    is_enabled: bool = Field(default=True)
    start_at: int
    end_at: int


class Chunk(ChunkBase, table=True):
    """文档分块表模型"""

    __tablename__ = "chunks"

    id: str = Field(default=None, primary_key=True, max_length=36)
    seq_id: int | None = None
    tenant_id: int
    knowledge_base_id: str = Field(max_length=36)
    knowledge_id: str = Field(max_length=36)

    pre_chunk_id: str | None = Field(default=None, max_length=36)
    next_chunk_id: str | None = Field(default=None, max_length=36)
    chunk_type: str = Field(default="text", max_length=20)  # text, document, faq, image
    parent_chunk_id: str | None = Field(default=None, max_length=36)

    tag_id: str | None = Field(default=None, max_length=36)
    image_info: str | None = None
    meta_data: Any | None = Field(default=None, sa_column=Column(JSONB))
    relation_chunks: Any | None = Field(default=None, sa_column=Column(JSONB))
    indirect_relation_chunks: Any | None = Field(default=None, sa_column=Column(JSONB))
    content_hash: str | None = Field(default=None, max_length=64)
    flags: int = Field(default=1)
    status: int = Field(default=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class ChunkCreate(ChunkBase):
    """文档分块创建模型"""

    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str


# ============== Embedding (向量) ==============


class EmbeddingBase(SQLModel):
    """向量基础模型"""

    source_id: str = Field(max_length=64)
    source_type: int  # 1: chunk, 2: knowledge
    dimension: int


class Embedding(EmbeddingBase, table=True):
    """向量表模型"""

    __tablename__ = "embeddings"

    id: int | None = Field(default=None, primary_key=True)
    chunk_id: str | None = Field(default=None, max_length=64)
    knowledge_id: str | None = Field(default=None, max_length=64)
    knowledge_base_id: str | None = Field(default=None, max_length=64)
    content: str | None = None
    # embedding 字段需要在实际使用时通过原始 SQL 处理
    # 这里简化为字符串存储
    embedding_value: str | None = Field(default=None, max_length=10000)
    is_enabled: bool = Field(default=True)
    tag_id: str | None = Field(default=None, max_length=36)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============== KnowledgeTag (知识标签) ==============


class KnowledgeTagBase(SQLModel):
    """知识标签基础模型"""

    name: str = Field(max_length=128)
    color: str | None = Field(default=None, max_length=32)
    sort_order: int = Field(default=0)


class KnowledgeTag(KnowledgeTagBase, table=True):
    """知识标签表模型"""

    __tablename__ = "knowledge_tags"

    id: str = Field(default=None, primary_key=True, max_length=36)
    seq_id: int | None = None
    tenant_id: int
    knowledge_base_id: str = Field(max_length=36)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class KnowledgeTagCreate(KnowledgeTagBase):
    """知识标签创建模型"""

    tenant_id: int
    knowledge_base_id: str
