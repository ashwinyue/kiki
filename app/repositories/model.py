"""模型 Repository

提供模型的数据访问操作
"""

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Model
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class ModelRepository(BaseRepository[Model]):
    """模型仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Model, session)

    async def create_with_tenant(
        self, data: dict, tenant_id: int
    ) -> Model:
        """创建模型

        Args:
            data: 模型数据
            tenant_id: 租户 ID

        Returns:
            创建的模型实例
        """
        model = Model(
            id=str(uuid4()),
            tenant_id=tenant_id,
            **data,
        )
        return await self.create(model)

    async def list_by_type(
        self,
        model_type: str | None,
        tenant_id: int,
        params: PaginationParams,
    ) -> PaginatedResult[Model]:
        """按类型获取模型列表

        Args:
            model_type: 模型类型
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        if model_type:
            return await self.list_paginated_by_tenant(
                tenant_id, params, type=model_type
            )
        return await self.list_paginated_by_tenant(tenant_id, params)

    async def get_default(
        self, model_type: str, tenant_id: int
    ) -> Model | None:
        """获取默认模型

        Args:
            model_type: 模型类型
            tenant_id: 租户 ID

        Returns:
            默认模型实例
        """
        return await self.get_by(
            tenant_id=tenant_id,
            type=model_type,
            is_default=True,
            deleted_at=None,
        )

    async def set_default(
        self, model_id: str, tenant_id: int
    ) -> bool:
        """设置默认模型（取消同类型的其他默认）

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            是否设置成功
        """
        # 获取模型
        model = await self.get_by_tenant(model_id, tenant_id)
        if not model:
            return False

        # 取消同类型的其他默认
        stmt = select(Model).where(
            Model.tenant_id == tenant_id,
            Model.type == model.type,
            Model.id != model_id,
            Model.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        for m in result.scalars():
            m.is_default = False

        # 设置新的默认
        model.is_default = True
        await self.session.commit()
        return True

    async def soft_delete(self, model_id: str, tenant_id: int) -> bool:
        """软删除模型

        Args:
            model_id: 模型 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        from datetime import UTC, datetime

        model = await self.get_by_tenant(model_id, tenant_id)
        if model:
            model.deleted_at = datetime.now(UTC)
            model.is_default = False
            await self.session.commit()
            return True
        return False


__all__ = ["ModelRepository"]
