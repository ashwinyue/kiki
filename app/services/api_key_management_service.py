"""API Key 管理服务层

提供 API Key 的 CRUD 业务逻辑，包括创建、查询、更新、删除、统计等。
封装响应构建和权限检查逻辑，消除路由层的重复代码。
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api_key import ApiKeyService as CoreApiKeyService
from app.infra.database import get_session
from app.models.api_key import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyRead,
    ApiKeyResponse,
    ApiKeyType,
    ApiKeyUpdate,
)
from app.observability.logging import get_logger
from app.repositories.api_key import ApiKeyRepository

logger = get_logger(__name__)


class ApiKeyManagementService:
    """API Key 管理服务

    封装 API Key 的业务逻辑，包括：
    - 创建 API Key（生成、哈希、存储）
    - 查询和列表
    - 更新和删除
    - 权限验证
    - 统计信息
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._repository: ApiKeyRepository | None = None

    @property
    def repository(self) -> ApiKeyRepository:
        """获取仓储（延迟初始化）"""
        if self._repository is None:
            self._repository = ApiKeyRepository(self.session)
        return self._repository

    # ============== 响应构建方法 ==============

    def _to_response(self, api_key: ApiKey, full_key: str | None = None) -> ApiKeyResponse:
        """转换为 API Key 响应

        Args:
            api_key: API Key 模型
            full_key: 完整的 Key（仅创建时提供）

        Returns:
            ApiKeyResponse 响应对象
        """
        return ApiKeyResponse(
            id=api_key.id,
            name=api_key.name,
            key=full_key if full_key else "",
            key_prefix=api_key.key_prefix,
            key_type=api_key.key_type,
            status=api_key.status,
            scopes=api_key.scopes or [],
            expires_at=api_key.expires_at,
            created_at=api_key.created_at,
        )

    def _to_read(self, api_key: ApiKey) -> ApiKeyRead:
        """转换为 API Key 读取模型

        Args:
            api_key: API Key 模型

        Returns:
            ApiKeyRead 读取模型
        """
        return ApiKeyRead(
            id=api_key.id,
            name=api_key.name,
            key_prefix=api_key.key_prefix,
            key_type=api_key.key_type,
            status=api_key.status,
            scopes=api_key.scopes or [],
            expires_at=api_key.expires_at,
            last_used_at=api_key.last_used_at,
            created_at=api_key.created_at,
            updated_at=api_key.updated_at,
        )

    # ============== 权限验证 ==============

    async def _verify_ownership(self, api_key_id: int, user_id: int) -> ApiKey:
        """验证 API Key 归属

        Args:
            api_key_id: API Key ID
            user_id: 用户 ID

        Returns:
            API Key 模型

        Raises:
            HTTPException: Key 不存在或无权限时
        """
        api_key = await self.repository.get_by_id(api_key_id)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API Key not found",
            )

        if api_key.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        return api_key

    # ============== CRUD 操作 ==============

    async def create_api_key(
        self,
        data: ApiKeyCreate,
        user_id: int,
    ) -> ApiKeyResponse:
        """创建 API Key

        生成完整的 API Key，哈希后存储，返回响应。

        Args:
            data: 创建数据
            user_id: 用户 ID

        Returns:
            ApiKeyResponse 包含完整 Key（仅此一次）
        """
        # 生成 API Key
        full_key, key_prefix = CoreApiKeyService.generate_key(data.key_type)

        # 哈希 Key
        hashed_key = CoreApiKeyService.hash_key(full_key)

        # 创建记录
        api_key = await self.repository.create_with_hash(
            data=data,
            user_id=user_id,
            hashed_key=hashed_key,
            key_prefix=key_prefix,
        )

        logger.info(
            "api_key_created",
            api_key_id=api_key.id,
            user_id=user_id,
            key_type=data.key_type.value,
        )

        return self._to_response(api_key, full_key)

    async def list_api_keys(
        self,
        user_id: int,
        status_filter: Any = None,
        key_type: Any = None,
    ) -> list[ApiKeyRead]:
        """列出用户的 API Key

        Args:
            user_id: 用户 ID
            status_filter: 状态筛选
            key_type: 类型筛选

        Returns:
            ApiKeyRead 列表
        """
        api_keys = await self.repository.list_by_user(
            user_id=user_id,
            status=status_filter,
            key_type=key_type,
        )

        return [self._to_read(key) for key in api_keys]

    async def get_api_key(self, api_key_id: int, user_id: int) -> ApiKeyRead:
        """获取 API Key 详情

        Args:
            api_key_id: API Key ID
            user_id: 用户 ID

        Returns:
            ApiKeyRead 读取模型

        Raises:
            HTTPException: Key 不存在或无权限时
        """
        api_key = await self._verify_ownership(api_key_id, user_id)
        return self._to_read(api_key)

    async def update_api_key(
        self,
        api_key_id: int,
        user_id: int,
        data: ApiKeyUpdate,
    ) -> ApiKeyRead:
        """更新 API Key

        Args:
            api_key_id: API Key ID
            user_id: 用户 ID
            data: 更新数据

        Returns:
            更新后的 ApiKeyRead

        Raises:
            HTTPException: Key 不存在或无权限时
        """
        api_key = await self._verify_ownership(api_key_id, user_id)

        # 更新字段
        if data.name is not None:
            api_key.name = data.name
        if data.status is not None:
            api_key.status = data.status
        if data.scopes is not None:
            api_key.scopes = data.scopes
        if data.expires_at is not None:
            api_key.expires_at = data.expires_at

        await self.session.commit()
        await self.session.refresh(api_key)

        logger.info("api_key_updated", api_key_id=api_key.id, user_id=user_id)

        return self._to_read(api_key)

    async def delete_api_key(self, api_key_id: int, user_id: int) -> None:
        """删除 API Key（软删除）

        Args:
            api_key_id: API Key ID
            user_id: 用户 ID

        Raises:
            HTTPException: Key 不存在、无权限或删除失败时
        """
        # 先验证权限
        await self._verify_ownership(api_key_id, user_id)

        success = await self.repository.soft_delete(api_key_id)

        if success:
            logger.info("api_key_deleted", api_key_id=api_key_id, user_id=user_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete API Key",
            )

    async def revoke_api_key(self, api_key_id: int, user_id: int) -> ApiKeyRead:
        """吊销 API Key

        Args:
            api_key_id: API Key ID
            user_id: 用户 ID

        Returns:
            吊销后的 ApiKeyRead

        Raises:
            HTTPException: Key 不存在或无权限时
        """
        # 验证权限（已包含存在性检查）
        await self._verify_ownership(api_key_id, user_id)

        revoked = await self.repository.revoke(api_key_id)

        if revoked:
            logger.info("api_key_revoked", api_key_id=revoked.id, user_id=user_id)
            return self._to_read(revoked)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API Key",
        )

    # ============== 统计方法 ==============

    async def get_stats(self, user_id: int) -> dict[str, Any]:
        """获取 API Key 统计

        Args:
            user_id: 用户 ID

        Returns:
            统计信息字典
        """
        # 按状态统计
        by_status = await self.repository.count_by_user(user_id)

        # 获取所有 Key 用于按类型统计
        all_keys = await self.repository.list_by_user(user_id)

        # 按类型统计
        type_counts: dict[str, int] = {}
        for key in all_keys:
            key_type = key.key_type.value
            type_counts[key_type] = type_counts.get(key_type, 0) + 1

        return {
            "user_id": user_id,
            "total_keys": len(all_keys),
            "by_status": by_status,
            "by_type": type_counts,
        }

    # ============== MCP 专用方法 ==============

    async def create_mcp_api_key(
        self,
        user_id: int,
        name: str,
        expires_in_days: int | None = None,
    ) -> ApiKeyResponse:
        """创建 MCP 专用 API Key

        Args:
            user_id: 用户 ID
            name: Key 名称
            expires_in_days: 有效期（天数）

        Returns:
            ApiKeyResponse 包含完整 Key
        """
        data = ApiKeyCreate(
            name=name,
            key_type=ApiKeyType.MCP,
            scopes=["chat", "agents:read", "tools:read"],
            expires_in_days=expires_in_days,
            description="MCP server access key",
        )

        return await self.create_api_key(data, user_id)


# ============== 依赖注入工厂 ==============


def get_api_key_management_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ApiKeyManagementService:
    """创建 API Key 管理服务实例

    Args:
        session: 数据库会话

    Returns:
        ApiKeyManagementService 实例
    """
    return ApiKeyManagementService(session)
