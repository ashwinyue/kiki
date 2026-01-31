"""租户仓储

提供租户配置的数据访问操作。
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Tenant, TenantCreate, TenantUpdate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class TenantRepository(BaseRepository[Tenant]):
    """租户仓储类"""

    def __init__(self, session: AsyncSession):
        super().__init__(Tenant, session)

    async def create_with_config(self, data: TenantCreate) -> Tenant:
        """创建租户

        Args:
            data: 租户创建数据

        Returns:
            创建的租户
        """
        tenant = Tenant(**data.model_dump())
        return await self.create(tenant)

    async def get_by_api_key(self, api_key: str) -> Tenant | None:
        """根据 API Key 获取租户

        Args:
            api_key: 租户 API Key

        Returns:
            租户实例或 None
        """
        try:
            statement = select(Tenant).where(
                Tenant.api_key == api_key,
                Tenant.deleted_at.is_(None),  # type: ignore
            )
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("tenant_repository_get_by_api_key_failed", api_key=api_key[:8], error=str(e))
            return None

    async def get_by_name(self, name: str) -> Tenant | None:
        """根据名称获取租户

        Args:
            name: 租户名称

        Returns:
            租户实例或 None
        """
        try:
            statement = select(Tenant).where(
                Tenant.name == name,
                Tenant.deleted_at.is_(None),  # type: ignore
            )
            result = await self.session.execute(statement)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("tenant_repository_get_by_name_failed", name=name, error=str(e))
            return None

    async def list_by_status(
        self,
        status: str = "active",
        params: PaginationParams | None = None,
    ) -> PaginatedResult[Tenant]:
        """按状态列出租户

        Args:
            status: 租户状态
            params: 分页参数

        Returns:
            分页结果
        """
        try:
            statement = select(Tenant).where(
                Tenant.status == status,
                Tenant.deleted_at.is_(None),  # type: ignore
            )

            # 获取总数
            from sqlalchemy import func

            count_stmt = select(func.count()).select_from(statement.subquery())
            total_result = await self.session.execute(count_stmt)
            total = total_result.scalar() or 0

            # 分页
            if params is None:
                params = PaginationParams()
            statement = statement.offset(params.offset).limit(params.limit)

            items_result = await self.session.execute(statement)
            items = list(items_result.scalars().all())

            return PaginatedResult.create(items, total, params)

        except Exception as e:
            logger.error("tenant_repository_list_by_status_failed", status=status, error=str(e))
            return PaginatedResult.create([], 0, params or PaginationParams())

    async def update_tenant(
        self,
        tenant_id: int,
        data: TenantUpdate,
    ) -> Tenant | None:
        """更新租户

        Args:
            tenant_id: 租户 ID
            data: 更新数据

        Returns:
            更新后的租户或 None
        """
        try:
            tenant = await self.get(tenant_id)
            if tenant is None:
                return None

            update_data = data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(tenant, field):
                    setattr(tenant, field, value)

            await self.session.commit()
            await self.session.refresh(tenant)

            logger.info("tenant_updated", tenant_id=tenant_id)
            return tenant

        except Exception as e:
            await self.session.rollback()
            logger.error("tenant_repository_update_failed", tenant_id=tenant_id, error=str(e))
            raise

    async def soft_delete(self, tenant_id: int) -> bool:
        """软删除租户

        Args:
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        try:
            tenant = await self.get(tenant_id)
            if tenant is None:
                return False

            from datetime import UTC, datetime

            tenant.deleted_at = datetime.now(UTC)
            await self.session.commit()

            logger.info("tenant_soft_deleted", tenant_id=tenant_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("tenant_repository_delete_failed", tenant_id=tenant_id, error=str(e))
            return False

    async def get_active_count(self) -> int:
        """获取活跃租户数量

        Returns:
            活跃租户数量
        """
        try:
            from sqlalchemy import func

            statement = select(func.count()).select_from(Tenant).where(
                Tenant.status == "active",
                Tenant.deleted_at.is_(None),  # type: ignore
            )
            result = await self.session.execute(statement)
            return result.scalar() or 0

        except Exception as e:
            logger.error("tenant_repository_count_failed", error=str(e))
            return 0

    async def update_config(
        self,
        tenant_id: int,
        config: dict[str, Any],
    ) -> Tenant | None:
        """更新租户配置

        Args:
            tenant_id: 租户 ID
            config: 配置字典

        Returns:
            更新后的租户或 None
        """
        try:
            tenant = await self.get(tenant_id)
            if tenant is None:
                return None

            # 合并配置
            existing_config = tenant.config or {}
            existing_config.update(config)
            tenant.config = existing_config

            await self.session.commit()
            await self.session.refresh(tenant)

            logger.info("tenant_config_updated", tenant_id=tenant_id)
            return tenant

        except Exception as e:
            await self.session.rollback()
            logger.error("tenant_repository_update_config_failed", tenant_id=tenant_id, error=str(e))
            raise
