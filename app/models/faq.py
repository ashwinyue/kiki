"""FAQ 模型

对齐 WeKnora99 表结构，提供 FAQ（常见问题）管理功能。
"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlmodel import Column, Field, SQLModel


# ============== 枚举类型 ==============


class FAQCategory(str, Enum):
    """FAQ 分类"""

    GENERAL = "general"  # 通用问题
    ACCOUNT = "account"  # 账号相关
    BILLING = "billing"  # 计费相关
    TECHNICAL = "technical"  # 技术问题
    API = "api"  # API 使用
    INTEGRATION = "integration"  # 集成问题
    SECURITY = "security"  # 安全问题
    OTHER = "other"  # 其他


class FAQStatus(str, Enum):
    """FAQ 状态"""

    DRAFT = "draft"  # 草稿
    PUBLISHED = "published"  # 已发布
    ARCHIVED = "archived"  # 已归档


# ============== 数据库模型 ==============


class FAQBase(SQLModel):
    """FAQ 基础模型"""

    question: str = Field(max_length=500, description="问题")
    answer: str = Field(sa_column=Column(Text), description="答案")
    category: FAQCategory = Field(default=FAQCategory.GENERAL, description="分类")
    tags: list[str] | None = Field(
        default=None,
        sa_column=Column(ARRAY(String), default=None),
        description="标签",
    )
    priority: int = Field(default=0, description="优先级（数字越大越靠前）")
    locale: str = Field(default="zh-CN", max_length=10, description="语言")


class FAQ(FAQBase, table=True):
    """FAQ 表模型"""

    __tablename__ = "faqs"

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: int | None = Field(
        default=None,
        foreign_key="tenants.id",
        index=True,
        description="租户 ID（null 表示全局 FAQ）",
    )
    status: FAQStatus = Field(default=FAQStatus.DRAFT, description="状态")
    slug: str | None = Field(
        default=None,
        max_length=200,
        index=True,
        description="URL 友好的标识符",
    )
    view_count: int = Field(default=0, description="浏览次数")
    helpful_count: int = Field(default=0, description="有用点赞数")
    not_helpful_count: int = Field(default=0, description="无用点踩数")
    search_vector: str | None = Field(
        default=None,
        sa_column=Column(TSVECTOR, default=None),
        description="全文搜索向量",
    )
    created_by: int | None = Field(
        default=None,
        foreign_key="users.id",
        description="创建人 ID",
    )
    updated_by: int | None = Field(
        default=None,
        foreign_key="users.id",
        description="更新人 ID",
    )
    published_at: datetime | None = Field(default=None, description="发布时间")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============== 创建/更新模型 ==============


class FAQCreate(SQLModel):
    """FAQ 创建模型"""

    question: str = Field(max_length=500, description="问题")
    answer: str = Field(description="答案")
    category: FAQCategory = Field(default=FAQCategory.GENERAL, description="分类")
    tags: list[str] | None = Field(default=None, description="标签")
    priority: int = Field(default=0, description="优先级")
    locale: str = Field(default="zh-CN", max_length=10, description="语言")
    slug: str | None = Field(default=None, max_length=200, description="URL 标识符")


class FAQUpdate(SQLModel):
    """FAQ 更新模型"""

    question: str | None = None
    answer: str | None = None
    category: FAQCategory | None = None
    tags: list[str] | None = None
    priority: int | None = None
    status: FAQStatus | None = None
    locale: str | None = None
    slug: str | None = None


# ============== 响应模型 ==============


class FAQRead(SQLModel):
    """FAQ 读取模型"""

    id: int
    question: str
    answer: str
    category: FAQCategory
    tags: list[str] | None
    priority: int
    locale: str
    status: FAQStatus
    slug: str | None
    view_count: int
    helpful_count: int
    not_helpful_count: int
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None


class FAQDetail(FAQRead):
    """FAQ 详情模型（包含更多字段）"""

    tenant_id: int | None
    created_by: int | None
    updated_by: int | None


class FAQSearchResult(SQLModel):
    """FAQ 搜索结果"""

    id: int
    question: str
    answer: str
    category: FAQCategory
    tags: list[str] | None
    locale: str
    slug: str | None
    relevance_score: float | None = Field(default=None, description="相关性评分")
    rank: int | None = Field(default=None, description="排名")


# ============== 批量操作模型 ==============


class FAQBulkUpdate(SQLModel):
    """FAQ 批量更新模型"""

    ids: list[int] = Field(description="要更新的 FAQ ID 列表")
    status: FAQStatus | None = None
    category: FAQCategory | None = None
    priority: int | None = None


class FAQReorderRequest(SQLModel):
    """FAQ 重新排序请求"""

    id_orders: dict[int, int] = Field(
        description="FAQ ID 到优先级的映射，如 {1: 10, 2: 20, 3: 30}"
    )


# ============== 反馈模型 ==============


class FAQFeedbackCreate(SQLModel):
    """FAQ 反馈创建模型"""

    helpful: bool = Field(description="是否有用")
