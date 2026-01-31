"""知识标签 Repository

提供知识标签的数据访问操作
"""

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Chunk, Knowledge, KnowledgeTag
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class TagRepository(BaseRepository[KnowledgeTag]):
    """知识标签仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(KnowledgeTag, session)

    async def create_with_tenant(
        self, data: dict, tenant_id: int
    ) -> KnowledgeTag:
        """创建标签并生成 ID

        Args:
            data: 标签数据
            tenant_id: 租户 ID

        Returns:
            创建的标签实例
        """
        tag = KnowledgeTag(
            id=str(uuid4()),
            tenant_id=tenant_id,
            **data,
        )
        return await self.create(tag)

    async def list_by_kb(
        self,
        kb_id: str,
        tenant_id: int,
        params: PaginationParams,
        keyword: str | None = None,
    ) -> PaginatedResult[KnowledgeTag]:
        """按知识库分页获取标签

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 分页参数
            keyword: 关键词搜索

        Returns:
            分页结果
        """
        stmt = select(KnowledgeTag).where(
            KnowledgeTag.knowledge_base_id == kb_id,
            KnowledgeTag.tenant_id == tenant_id,
            KnowledgeTag.deleted_at.is_(None),
        )

        # 关键词搜索
        if keyword:
            stmt = stmt.where(KnowledgeTag.name.ilike(f"%{keyword}%"))

        # 排序
        stmt = stmt.order_by(KnowledgeTag.sort_order, KnowledgeTag.created_at)

        # 分页
        stmt = stmt.limit(params.size).offset((params.page - 1) * params.size)

        result = await self.session.execute(stmt)
        items = result.scalars().all()

        # 获取总数
        count_stmt = select(func.count()).select_from(KnowledgeTag).where(
            KnowledgeTag.knowledge_base_id == kb_id,
            KnowledgeTag.tenant_id == tenant_id,
            KnowledgeTag.deleted_at.is_(None),
        )
        if keyword:
            count_stmt = count_stmt.where(KnowledgeTag.name.ilike(f"%{keyword}%"))

        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar() or 0

        return PaginatedResult(
            items=list(items),
            total=total,
            page=params.page,
            size=params.size,
        )

    async def soft_delete(self, tag_id: str, tenant_id: int) -> bool:
        """软删除标签

        Args:
            tag_id: 标签 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        from datetime import UTC, datetime

        tag = await self.get_by_tenant(tag_id, tenant_id)
        if tag:
            tag.deleted_at = datetime.now(UTC)
            await self.session.commit()
            return True
        return False

    async def get_knowledge_count(self, tag_id: str, tenant_id: int) -> int:
        """获取标签关联的知识数量

        Args:
            tag_id: 标签 ID
            tenant_id: 租户 ID

        Returns:
            知识数量
        """
        stmt = select(func.count()).select_from(Knowledge).where(
            Knowledge.tag_id == tag_id,
            Knowledge.tenant_id == tenant_id,
            Knowledge.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_chunk_count(self, tag_id: str, tenant_id: int) -> int:
        """获取标签关联的分块数量

        Args:
            tag_id: 标签 ID
            tenant_id: 租户 ID

        Returns:
            分块数量
        """
        stmt = select(func.count()).select_from(Chunk).where(
            Chunk.tag_id == tag_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0


__all__ = ["TagRepository"]
