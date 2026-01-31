"""API Key 数据仓储

提供 API Key 的数据访问操作。
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyStatus,
    ApiKeyType,
)
from app.repositories.base import BaseRepository


class ApiKeyRepository(BaseRepository[ApiKey]):
    """API Key 仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ApiKey, session)

    async def create_with_hash(
        self,
        data: ApiKeyCreate,
        user_id: int,
        hashed_key: str,
        key_prefix: str,
    ) -> ApiKey:
        """创建 API Key（带哈希）

        Args:
            data: 创建数据
            user_id: 用户 ID
            hashed_key: 哈希后的 Key
            key_prefix: Key 前缀

        Returns:
            创建的 API Key
        """

        # 计算过期时间
        expires_at = None
        if data.expires_in_days:
            expires_at = datetime.now(UTC) + timedelta(days=data.expires_in_days)

        api_key = ApiKey(
            name=data.name,
            key_prefix=key_prefix,
            key_type=data.key_type,
            status=ApiKeyStatus.ACTIVE,
            user_id=user_id,
            scopes=data.scopes,
            expires_at=expires_at,
            hashed_key=hashed_key,
            description=data.description,
            rate_limit=data.rate_limit,
        )

        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)

        return api_key

    async def get_by_id(self, api_key_id: int) -> ApiKey | None:
        """根据 ID 获取 API Key

        Args:
            api_key_id: API Key ID

        Returns:
            API Key 对象
        """
        return await self.get(api_key_id)

    async def list_by_user(
        self,
        user_id: int,
        status: ApiKeyStatus | None = None,
        key_type: ApiKeyType | None = None,
    ) -> list[ApiKey]:
        """列出用户的 API Key

        Args:
            user_id: 用户 ID
            status: 状态筛选
            key_type: 类型筛选

        Returns:
            API Key 列表
        """
        statement = select(ApiKey).where(ApiKey.user_id == user_id)

        if status:
            statement = statement.where(ApiKey.status == status)

        if key_type:
            statement = statement.where(ApiKey.key_type == key_type)

        statement = statement.order_by(ApiKey.created_at.desc())

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_prefix(self, prefix: str) -> list[ApiKey]:
        """根据前缀查找 API Key

        Args:
            prefix: Key 前缀

        Returns:
            API Key 列表
        """
        statement = select(ApiKey).where(
            ApiKey.key_prefix == prefix,
            ApiKey.status == ApiKeyStatus.ACTIVE,
        )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_type(
        self,
        key_type: ApiKeyType,
        status: ApiKeyStatus | None = None,
    ) -> list[ApiKey]:
        """根据类型列出 API Key

        Args:
            key_type: Key 类型
            status: 状态筛选

        Returns:
            API Key 列表
        """
        statement = select(ApiKey).where(ApiKey.key_type == key_type)

        if status:
            statement = statement.where(ApiKey.status == status)

        statement = statement.order_by(ApiKey.created_at.desc())

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update_status(
        self,
        api_key_id: int,
        status: ApiKeyStatus,
    ) -> ApiKey | None:
        """更新 API Key 状态

        Args:
            api_key_id: API Key ID
            status: 新状态

        Returns:
            更新后的 API Key
        """
        api_key = await self.get(api_key_id)
        if api_key:
            api_key.status = status
            await self.session.commit()
            await self.session.refresh(api_key)
        return api_key

    async def update_last_used(self, api_key_id: int) -> None:
        """更新最后使用时间

        Args:
            api_key_id: API Key ID
        """
        api_key = await self.get(api_key_id)
        if api_key:
            api_key.last_used_at = datetime.now(UTC)
            await self.session.commit()

    async def revoke(self, api_key_id: int) -> ApiKey | None:
        """吊销 API Key

        Args:
            api_key_id: API Key ID

        Returns:
            更新后的 API Key
        """
        return await self.update_status(api_key_id, ApiKeyStatus.REVOKED)

    async def soft_delete(self, api_key_id: int) -> bool:
        """软删除 API Key

        Args:
            api_key_id: API Key ID

        Returns:
            是否成功
        """
        api_key = await self.get(api_key_id)
        if api_key:
            # 标记为已吊销而不是真删除
            api_key.status = ApiKeyStatus.REVOKED
            await self.session.commit()
            return True
        return False

    async def count_by_user(self, user_id: int) -> dict[str, int]:
        """统计用户的 API Key 数量

        Args:
            user_id: 用户 ID

        Returns:
            统计结果
        """
        from sqlalchemy import func

        statement = (
            select(ApiKey.status, func.count(ApiKey.id))
            .where(ApiKey.user_id == user_id)
            .group_by(ApiKey.status)
        )

        result = await self.session.execute(statement)
        return {status.value: count for status, count in result.all()}

    async def cleanup_expired(self, days: int = 30) -> int:
        """清理过期的 API Key

        Args:
            days: 过期天数阈值

        Returns:
            清理的数量
        """
        from datetime import timedelta

        threshold = datetime.now(UTC) - timedelta(days=days)

        statement = select(ApiKey).where(
            ApiKey.status == ApiKeyStatus.EXPIRED,
            ApiKey.updated_at < threshold,
        )

        result = await self.session.execute(statement)
        keys = result.scalars().all()

        count = 0
        for key in keys:
            await self.session.delete(key)
            count += 1

        await self.session.commit()
        return count
