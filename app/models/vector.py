"""向量相关数据模型

定义向量存储相关的 SQLModel 模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, SQLModel
from sqlmodel import Field as SQLField

# ============== 向量存储配置模型 ==============


class VectorStoreConfigModel(BaseModel):
    """向量存储配置模型

    Attributes:
        store_type: 存储类型
        collection_name: 集合名称
        dimension: 向量维度
        metric: 相似度度量
    """

    store_type: str = "memory"
    collection_name: str = "default"
    dimension: int = 1024
    metric: str = "cosine"


# ============== 文档向量模型 ==============


class DocumentVectorBase(SQLModel):
    """文档向量基础模型"""

    doc_id: str = SQLField(max_length=255, index=True)
    collection_name: str = SQLField(max_length=255, index=True, default="default")
    content: str = SQLField(default="")
    metadata: Any | None = SQLField(default=None, sa_column=Column(JSONB))
    tenant_id: int | None = SQLField(default=None, index=True)


class DocumentVector(DocumentVectorBase, table=True):
    """文档向量表模型

    用于追踪已索引的文档，支持重复检测和更新管理。
    注意：实际向量数据存储在向量数据库中，此表仅用于元数据追踪。
    """

    __tablename__ = "document_vectors"

    id: int | None = SQLField(default=None, primary_key=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    indexed_at: datetime | None = SQLField(default=None)  # 最后索引时间
    vector_count: int = SQLField(default=0)  # 包含的向量数量


class DocumentVectorCreate(SQLModel):
    """文档向量创建模型"""

    doc_id: str
    collection_name: str = "default"
    content: str
    metadata: Any | None = None
    tenant_id: int | None = None


class DocumentVectorUpdate(SQLModel):
    """文档向量更新模型"""

    content: str | None = None
    metadata: Any | None = None
    indexed_at: datetime | None = None
    vector_count: int | None = None


class DocumentVectorPublic(DocumentVectorBase):
    """文档向量公开信息"""

    id: int
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None
    vector_count: int


# ============== 向量搜索历史模型 ==============


class VectorSearchHistoryBase(SQLModel):
    """向量搜索历史基础模型"""

    query: str = SQLField(default="")
    collection_name: str = SQLField(max_length=255, default="default")
    results_count: int = SQLField(default=0)
    top_k: int = SQLField(default=5)
    filters: Any | None = SQLField(default=None, sa_column=Column(JSONB))
    tenant_id: int | None = SQLField(default=None, index=True)
    user_id: int | None = SQLField(default=None, index=True)


class VectorSearchHistory(VectorSearchHistoryBase, table=True):
    """向量搜索历史表模型

    记录搜索请求，用于分析和优化。
    """

    __tablename__ = "vector_search_history"

    id: int | None = SQLField(default=None, primary_key=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    response_time_ms: int | None = SQLField(default=None)  # 响应时间（毫秒）


class VectorSearchHistoryCreate(SQLModel):
    """向量搜索历史创建模型"""

    query: str
    collection_name: str = "default"
    results_count: int = 0
    top_k: int = 5
    filters: Any | None = None
    tenant_id: int | None = None
    user_id: int | None = None
    response_time_ms: int | None = None


class VectorSearchHistoryPublic(VectorSearchHistoryBase):
    """向量搜索历史公开信息"""

    id: int
    created_at: datetime
    response_time_ms: int | None


__all__ = [
    "VectorStoreConfigModel",
    "DocumentVector",
    "DocumentVectorCreate",
    "DocumentVectorUpdate",
    "DocumentVectorPublic",
    "VectorSearchHistory",
    "VectorSearchHistoryCreate",
    "VectorSearchHistoryPublic",
]
