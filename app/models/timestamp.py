"""时间戳 Mixin - SQLAlchemy 最佳实践

使用 `server_default` + `onupdate` 实现自动更新时间戳，
无需数据库触发器，符合 SQLAlchemy 推荐模式。

参考: https://docs.sqlalchemy.org/en/20/core/defaults.html
"""

from datetime import datetime
from typing import ClassVar

from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import declared_attr
from sqlmodel import Field, SQLModel


class TimestampMixin(SQLModel):
    """时间戳 Mixin 基类

    为表提供自动管理的 `created_at` 和 `updated_at` 字段：

    - `created_at`: 插入时自动设置为当前时间
    - `updated_at`: 插入和更新时自动设置为当前时间

    使用方式：
        ```python
        class User(TimestampMixin, table=True):
            __tablename__ = "users"
            id: int | None = Field(default=None, primary_key=True)
            name: str
        ```

    注意事项：
        1. 必须作为第一个基类继承
        2. 不需要在子类中重复定义时间戳字段
        3. 数据库会自动管理这些字段的值
    """

    # 使用 ClassVar 标记为类变量，避免 Pydantic v2 类型检查错误
    created_at: ClassVar[datetime] = Field(
        sa_column=Column(
            DateTime,
            server_default=func.now(),
            nullable=False,
        )
    )

    updated_at: ClassVar[datetime] = Field(
        sa_column=Column(
            DateTime,
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )
    )


__all__ = ["TimestampMixin"]
