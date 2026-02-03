"""租户服务层

提供租户相关的业务逻辑。
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.tenant_api_key import generate_api_key
from app.models.database import Tenant, TenantCreate, TenantUpdate
from app.observability.logging import get_logger

logger = get_logger(__name__)


class TenantService:
    """租户服务"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_tenant(self, data: TenantCreate) -> Tenant:
        """创建租户

        Args:
            data: 租户创建数据

        Returns:
            创建的租户对象
        """
        logger.info("creating_tenant", name=data.name)

        tenant = Tenant(
            name=data.name,
            description=data.description,
            status=data.status or "active",
            config=data.config or {},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # 先保存获取 ID
        self.session.add(tenant)
        await self.session.flush()

        # 生成 API Key（需要租户 ID）
        tenant.api_key = generate_api_key(tenant.id)

        await self.session.commit()
        await self.session.refresh(tenant)

        logger.info("tenant_created", id=tenant.id, name=tenant.name)
        return tenant

    async def get_tenant_by_id(self, tenant_id: int) -> Tenant | None:
        """根据 ID 获取租户

        Args:
            tenant_id: 租户 ID

        Returns:
            租户对象，不存在返回 None
        """
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_tenant_by_api_key(self, api_key: str) -> Tenant | None:
        """根据 API Key 获取租户

        Args:
            api_key: API Key 字符串

        Returns:
            租户对象，不存在返回 None
        """
        stmt = select(Tenant).where(
            Tenant.api_key == api_key,
            Tenant.status == "active",
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_tenant_by_name(self, name: str) -> Tenant | None:
        """根据名称获取租户

        Args:
            name: 租户名称

        Returns:
            租户对象，不存在返回 None
        """
        stmt = select(Tenant).where(Tenant.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_tenant(
        self, tenant_id: int, data: TenantUpdate
    ) -> Tenant | None:
        """更新租户

        Args:
            tenant_id: 租户 ID
            data: 更新数据

        Returns:
            更新后的租户对象，不存在返回 None
        """
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        tenant.updated_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(tenant)

        logger.info("tenant_updated", id=tenant.id)
        return tenant

    async def rotate_api_key(self, tenant_id: int) -> str | None:
        """轮换 API Key

        Args:
            tenant_id: 租户 ID

        Returns:
            新的 API Key，租户不存在返回 None
        """
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            return None

        new_api_key = generate_api_key(tenant_id)
        tenant.api_key = new_api_key
        tenant.updated_at = datetime.now(UTC)

        await self.session.commit()

        logger.info("tenant_api_key_rotated", id=tenant_id)
        return new_api_key

    async def list_tenants(
        self,
        *,
        status: str | None = None,
        keyword: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Tenant], int]:
        """列出租户

        Args:
            status: 过滤状态
            keyword: 搜索关键词（名称或描述）
            offset: 偏移量
            limit: 限制数量

        Returns:
            (租户列表, 总数)
        """
        from sqlalchemy import func

        stmt = select(Tenant)
        count_stmt = select(func.count()).select_from(Tenant)

        if status:
            stmt = stmt.where(Tenant.status == status)
            count_stmt = count_stmt.where(Tenant.status == status)

        if keyword:
            pattern = f"%{keyword}%"
            stmt = stmt.where(
                (Tenant.name.ilike(pattern)) | (Tenant.description.ilike(pattern))
            )
            count_stmt = count_stmt.where(
                (Tenant.name.ilike(pattern)) | (Tenant.description.ilike(pattern))
            )

        # 获取总数
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        # 获取分页数据
        stmt = stmt.order_by(Tenant.id).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        tenants = list(result.scalars().all())

        return tenants, total

    async def delete_tenant(self, tenant_id: int) -> bool:
        """软删除租户

        Args:
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        tenant = await self.get_tenant_by_id(tenant_id)
        if not tenant:
            return False

        tenant.deleted_at = datetime.now(UTC)
        tenant.status = "deleted"
        tenant.updated_at = datetime.now(UTC)

        await self.session.commit()

        logger.info("tenant_deleted", id=tenant_id)
        return True

    async def check_api_key_exists(self, api_key: str) -> bool:
        """检查 API Key 是否已存在

        Args:
            api_key: API Key 字符串

        Returns:
            是否存在
        """
        stmt = select(Tenant).where(Tenant.api_key == api_key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def search_tenants(
        self,
        *,
        keyword: str | None = None,
        status: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[Tenant], int]:
        """搜索租户

        Args:
            keyword: 搜索关键词（名称、描述、业务类型）
            status: 状态筛选
            page: 页码
            size: 每页数量

        Returns:
            (租户列表, 总数)
        """
        from app.repositories.base import PaginationParams
        from app.repositories.tenant import TenantRepository

        repository = TenantRepository(self.session)
        params = PaginationParams(page=page, size=size)

        result = await repository.search(
            keyword=keyword,
            status=status,
            params=params,
        )

        logger.info(
            "tenant_service_search_success",
            keyword=keyword,
            status=status,
            total=result.total,
            page=params.page,
        )

        return result.items, result.total


# 解决循环导入
if __name__ == "__main__":
    pass
