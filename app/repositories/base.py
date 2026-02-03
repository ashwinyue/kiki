"""基础仓储

定义仓储层的基础接口和通用操作。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger

logger = get_logger(__name__)


class PaginationParams(BaseModel):
    """分页参数"""

    page: int = 1
    size: int = 20

    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        """返回限制数量"""
        return min(self.size, 100)  # 最大 100


class PaginatedResult[T](BaseModel):
    """分页结果"""

    items: list[T]
    total: int
    page: int
    size: int
    pages: int

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        params: PaginationParams,
    ) -> PaginatedResult[T]:
        """创建分页结果"""
        pages = (total + params.size - 1) // params.size if params.size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=params.page,
            size=params.size,
            pages=pages,
        )


class BaseRepository[T]:
    """基础仓储类

    提供通用的 CRUD 操作。
    """

    def __init__(self, model: type[T], session: AsyncSession):
        """初始化仓储

        Args:
            model: SQLModel 模型类
            session: 异步数据库会话
        """
        self.model = model
        self.session = session

    async def get(self, id: int | str) -> T | None:
        """根据 ID 获取单条记录

        Args:
            id: 记录 ID

        Returns:
            模型实例，不存在返回 None
        """
        try:
            statement = select(self.model).where(self.model.id == id)
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("repository_get_failed", model=self.model.__name__, id=id, error=str(e))
            return None

    async def get_by(
        self,
        **kwargs: Any,
    ) -> T | None:
        """根据条件获取单条记录

        Args:
            **kwargs: 查询条件

        Returns:
            模型实例，不存在返回 None
        """
        try:
            statement = select(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "repository_get_by_failed", model=self.model.__name__, kwargs=kwargs, error=str(e)
            )
            return None

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[T]:
        """获取记录列表

        Args:
            offset: 偏移量
            limit: 限制数量
            **kwargs: 过滤条件

        Returns:
            模型实例列表
        """
        try:
            statement = select(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)
            statement = statement.offset(offset).limit(limit)
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("repository_list_failed", model=self.model.__name__, error=str(e))
            return []

    async def list_paginated(
        self,
        params: PaginationParams,
        **kwargs: Any,
    ) -> PaginatedResult[T]:
        """分页获取记录列表

        Args:
            params: 分页参数
            **kwargs: 过滤条件

        Returns:
            分页结果
        """
        try:
            # 构建查询
            statement = select(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)

            # 获取总数
            count_statement = select(func.count()).select_from(statement.subquery())
            total_result = await self.session.execute(count_statement)
            total = total_result.scalar() or 0

            # 获取分页数据
            statement = statement.offset(params.offset).limit(params.limit)
            items_result = await self.session.execute(statement)
            items = list(items_result.scalars().all())

            return PaginatedResult.create(items, total, params)

        except Exception as e:
            logger.error(
                "repository_list_paginated_failed", model=self.model.__name__, error=str(e)
            )
            return PaginatedResult.create([], 0, params)

    async def create(self, obj: T) -> T:
        """创建记录

        Args:
            obj: 模型实例

        Returns:
            创建后的模型实例
        """
        try:
            self.session.add(obj)
            await self.session.commit()
            await self.session.refresh(obj)
            logger.info("repository_create_success", model=self.model.__name__, id=obj.id)
            return obj
        except Exception as e:
            await self.session.rollback()
            logger.error("repository_create_failed", model=self.model.__name__, error=str(e))
            raise

    async def update(
        self,
        id: int | str,
        **kwargs: Any,
    ) -> T | None:
        """更新记录

        Args:
            id: 记录 ID
            **kwargs: 更新字段

        Returns:
            更新后的模型实例，不存在返回 None
        """
        try:
            obj = await self.get(id)
            if obj is None:
                return None

            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)

            await self.session.commit()
            await self.session.refresh(obj)
            logger.info("repository_update_success", model=self.model.__name__, id=id)
            return obj

        except Exception as e:
            await self.session.rollback()
            logger.error("repository_update_failed", model=self.model.__name__, id=id, error=str(e))
            raise

    async def delete(self, id: int | str) -> bool:
        """删除记录

        Args:
            id: 记录 ID

        Returns:
            是否删除成功
        """
        try:
            obj = await self.get(id)
            if obj is None:
                return False

            await self.session.delete(obj)
            await self.session.commit()
            logger.info("repository_delete_success", model=self.model.__name__, id=id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("repository_delete_failed", model=self.model.__name__, id=id, error=str(e))
            raise

    async def count(self, **kwargs: Any) -> int:
        """统计记录数量

        Args:
            **kwargs: 过滤条件

        Returns:
            记录数量
        """
        try:
            statement = select(func.count()).select_from(self.model)
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)
            result = await self.session.execute(statement)
            return result.scalar() or 0
        except Exception as e:
            logger.error("repository_count_failed", model=self.model.__name__, error=str(e))
            return 0

    async def exists(self, **kwargs: Any) -> bool:
        """检查记录是否存在

        Args:
            **kwargs: 查询条件

        Returns:
            是否存在
        """
        count = await self.count(**kwargs)
        return count > 0

    # ============== 租户过滤方法 ==============

    async def get_by_tenant(
        self,
        id: int | str,
        tenant_id: int,
    ) -> T | None:
        """根据 ID 和租户 ID 获取记录

        自动添加租户过滤，确保数据隔离。

        Args:
            id: 记录 ID
            tenant_id: 租户 ID

        Returns:
            模型实例，不存在返回 None
        """
        try:
            # 检查模型是否有 tenant_id 字段
            if not hasattr(self.model, "tenant_id"):
                logger.warning(
                    "model_no_tenant_id",
                    model=self.model.__name__,
                )
                return await self.get(id)

            statement = select(self.model).where(
                self.model.id == id,  # type: ignore
                self.model.tenant_id == tenant_id,  # type: ignore
            )
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "repository_get_by_tenant_failed",
                model=self.model.__name__,
                id=id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return None

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        offset: int = 0,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[T]:
        """按租户获取记录列表

        自动添加租户过滤，确保数据隔离。

        Args:
            tenant_id: 租户 ID
            offset: 偏移量
            limit: 限制数量
            **kwargs: 额外的过滤条件

        Returns:
            模型实例列表
        """
        try:
            # 检查模型是否有 tenant_id 字段
            if not hasattr(self.model, "tenant_id"):
                logger.warning(
                    "model_no_tenant_id",
                    model=self.model.__name__,
                )
                return await self.list(offset=offset, limit=limit, **kwargs)

            statement = select(self.model).where(
                self.model.tenant_id == tenant_id,  # type: ignore
            )
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)
            statement = statement.offset(offset).limit(limit)
            result = await self.session.execute(statement)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(
                "repository_list_by_tenant_failed",
                model=self.model.__name__,
                tenant_id=tenant_id,
                error=str(e),
            )
            return []

    async def list_paginated_by_tenant(
        self,
        tenant_id: int,
        params: PaginationParams,
        **kwargs: Any,
    ) -> PaginatedResult[T]:
        """按租户分页获取记录列表

        自动添加租户过滤，确保数据隔离。

        Args:
            tenant_id: 租户 ID
            params: 分页参数
            **kwargs: 额外的过滤条件

        Returns:
            分页结果
        """
        try:
            # 检查模型是否有 tenant_id 字段
            if not hasattr(self.model, "tenant_id"):
                logger.warning(
                    "model_no_tenant_id",
                    model=self.model.__name__,
                )
                return await self.list_paginated(params, **kwargs)

            # 构建查询
            statement = select(self.model).where(
                self.model.tenant_id == tenant_id,  # type: ignore
            )
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    statement = statement.where(getattr(self.model, key) == value)

            # 获取总数
            count_stmt = select(func.count()).select_from(statement.subquery())
            total_result = await self.session.execute(count_stmt)
            total = total_result.scalar() or 0

            # 获取分页数据
            statement = statement.offset(params.offset).limit(params.limit)
            items_result = await self.session.execute(statement)
            items = list(items_result.scalars().all())

            return PaginatedResult.create(items, total, params)

        except Exception as e:
            logger.error(
                "repository_list_paginated_by_tenant_failed",
                model=self.model.__name__,
                tenant_id=tenant_id,
                error=str(e),
            )
            return PaginatedResult.create([], 0, params)
