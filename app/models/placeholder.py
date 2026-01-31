"""Agent 占位符模型

占位符是 Agent 配置中的可替换变量，支持模板渲染和变量验证。
对齐 WeKnora99 的 placeholders 功能。

示例：
    name: "company_name"
    description: "公司名称"
    default_value: "Kiki Inc"
    variable_type: "string"
    validation_rule: "^.{1,100}$"
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class PlaceholderBase(SQLModel):
    """占位符基础模型"""

    name: str = Field(max_length=100, description="变量名（用于模板中引用）")
    description: str | None = Field(None, max_length=500, description="变量描述")
    default_value: str | None = Field(None, max_length=2000, description="默认值")
    variable_type: str = Field(
        default="string",
        max_length=50,
        description="变量类型: string, number, boolean, json, array",
    )
    validation_rule: str | None = Field(
        None,
        max_length=500,
        description="验证规则（正则表达式或 JSON Schema）",
    )
    is_required: bool = Field(default=False, description="是否必填")
    is_enabled: bool = Field(default=True, description="是否启用")


class Placeholder(PlaceholderBase, table=True):
    """占位符表模型"""

    __tablename__ = "placeholders"

    id: str = Field(default=None, primary_key=True, max_length=36)
    tenant_id: int
    agent_id: str | None = Field(
        default=None,
        max_length=36,
        description="关联的 Agent ID（为空表示全局占位符）",
    )
    category: str | None = Field(
        default=None,
        max_length=100,
        description="分类（如: user, system, custom）",
    )
    display_order: int = Field(default=0, description="显示顺序")
    extra_metadata: Any | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="额外元数据",
    )
    created_by: str | None = Field(default=None, max_length=36)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = Field(default=None)


class PlaceholderCreate(PlaceholderBase):
    """占位符创建模型"""

    tenant_id: int
    agent_id: str | None = None
    category: str | None = None
    display_order: int | None = None
    extra_metadata: Any | None = None


class PlaceholderUpdate(SQLModel):
    """占位符更新模型"""

    name: str | None = None
    description: str | None = None
    default_value: str | None = None
    variable_type: str | None = None
    validation_rule: str | None = None
    is_required: bool | None = None
    is_enabled: bool | None = None
    category: str | None = None
    display_order: int | None = None
    extra_metadata: Any | None = None


class PlaceholderPublic(PlaceholderBase):
    """占位符公开信息"""

    id: str
    tenant_id: int
    agent_id: str | None
    category: str | None
    display_order: int
    created_at: datetime


# 导出别名
PlaceholderVar = Placeholder
PlaceholderVarCreate = PlaceholderCreate
PlaceholderVarUpdate = PlaceholderUpdate
PlaceholderVarPublic = PlaceholderPublic
