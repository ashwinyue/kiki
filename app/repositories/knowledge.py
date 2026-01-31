"""知识库 Repository

提供知识库、知识条目、文档分块的数据访问操作
"""

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Chunk, Knowledge, KnowledgeBase
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """知识库仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(KnowledgeBase, session)

    async def create_with_tenant(
        self, data: dict, tenant_id: int
    ) -> KnowledgeBase:
        """创建知识库并生成 ID

        Args:
            data: 知识库数据
            tenant_id: 租户 ID

        Returns:
            创建的知识库实例
        """
        kb = KnowledgeBase(
            id=str(uuid4()),
            tenant_id=tenant_id,
            **data,
        )
        return await self.create(kb)

    async def soft_delete(self, kb_id: str, tenant_id: int) -> bool:
        """软删除知识库

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        from datetime import UTC, datetime

        kb = await self.get_by_tenant(kb_id, tenant_id)
        if kb:
            kb.deleted_at = datetime.now(UTC)
            await self.session.commit()
            return True
        return False

    async def get_knowledge_count(self, kb_id: str, tenant_id: int) -> int:
        """获取知识库下的知识数量

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            知识条目数量
        """
        stmt = select(func.count()).select_from(Knowledge).where(
            Knowledge.knowledge_base_id == kb_id,
            Knowledge.tenant_id == tenant_id,
            Knowledge.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0


class KnowledgeRepository(BaseRepository[Knowledge]):
    """知识条目仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Knowledge, session)

    async def create_from_file(
        self,
        kb_id: str,
        tenant_id: int,
        file_name: str,
        file_type: str,
        file_size: int,
        file_path: str,
        title: str,
        **kwargs,
    ) -> Knowledge:
        """从文件创建知识条目

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            file_name: 文件名
            file_type: 文件类型
            file_size: 文件大小
            file_path: 文件路径
            title: 标题
            **kwargs: 额外字段

        Returns:
            创建的知识条目实例
        """
        knowledge = Knowledge(
            id=str(uuid4()),
            knowledge_base_id=kb_id,
            tenant_id=tenant_id,
            type="file",
            source="file_upload",
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            title=title,
            parse_status="processing",
            enable_status="enabled",
            **kwargs,
        )
        return await self.create(knowledge)

    async def create_from_url(
        self,
        kb_id: str,
        tenant_id: int,
        url: str,
        **kwargs,
    ) -> Knowledge:
        """从 URL 创建知识条目

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            url: URL 地址
            **kwargs: 额外字段

        Returns:
            创建的知识条目实例
        """
        knowledge = Knowledge(
            id=str(uuid4()),
            knowledge_base_id=kb_id,
            tenant_id=tenant_id,
            type="url",
            source=url,
            title=url,
            parse_status="processing",
            enable_status="enabled",
            **kwargs,
        )
        return await self.create(knowledge)

    async def create_manual(
        self,
        kb_id: str,
        tenant_id: int,
        title: str,
        content: str,
        **kwargs,
    ) -> Knowledge:
        """手工创建知识条目

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            title: 知识条目标题
            content: Markdown 内容
            **kwargs: 额外字段

        Returns:
            创建的知识条目实例
        """
        knowledge = Knowledge(
            id=str(uuid4()),
            knowledge_base_id=kb_id,
            tenant_id=tenant_id,
            type="text",
            source="manual",
            title=title,
            parse_status="processing",
            enable_status="enabled",
            **kwargs,
        )
        # 将内容存储在 meta_data 中
        if knowledge.meta_data is None:
            knowledge.meta_data = {}
        knowledge.meta_data["content"] = content

        return await self.create(knowledge)

    async def list_by_kb(
        self,
        kb_id: str,
        tenant_id: int,
        params: PaginationParams,
    ) -> PaginatedResult[Knowledge]:
        """按知识库分页获取知识条目

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated_by_tenant(
            tenant_id, params, knowledge_base_id=kb_id
        )

    async def soft_delete(self, knowledge_id: str, tenant_id: int) -> bool:
        """软删除知识条目

        Args:
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        from datetime import UTC, datetime

        knowledge = await self.get_by_tenant(knowledge_id, tenant_id)
        if knowledge:
            knowledge.deleted_at = datetime.now(UTC)
            await self.session.commit()
            return True
        return False

    async def update_chunk_count(
        self, knowledge_id: str, chunk_count: int
    ) -> None:
        """更新知识条目的分块数量

        Args:
            knowledge_id: 知识条目 ID
            chunk_count: 分块数量
        """
        stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
        result = await self.session.execute(stmt)
        knowledge = result.scalar_one_or_none()
        if knowledge:
            if knowledge.meta_data is None:
                knowledge.meta_data = {}
            knowledge.meta_data["chunk_count"] = chunk_count
            await self.session.commit()

    async def update_parse_status(
        self,
        knowledge_id: str,
        parse_status: str,
        error_message: str | None = None,
    ) -> None:
        """更新知识条目的解析状态

        Args:
            knowledge_id: 知识条目 ID
            parse_status: 解析状态
            error_message: 错误信息
        """
        from datetime import UTC, datetime

        stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
        result = await self.session.execute(stmt)
        knowledge = result.scalar_one_or_none()
        if knowledge:
            knowledge.parse_status = parse_status
            knowledge.error_message = error_message
            if parse_status == "completed":
                knowledge.processed_at = datetime.now(UTC)
            await self.session.commit()


class ChunkRepository(BaseRepository[Chunk]):
    """文档分块仓储"""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Chunk, session)

    async def list_by_knowledge(
        self,
        knowledge_id: str,
        tenant_id: int,
    ) -> list[Chunk]:
        """获取知识条目的所有分块

        Args:
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            分块列表
        """
        return await self.list_by_tenant(
            tenant_id,
            knowledge_id=knowledge_id,
            limit=10000,
        )

    async def create_chunks(
        self,
        chunks: list[dict],
        kb_id: str,
        knowledge_id: str,
        tenant_id: int,
    ) -> list[Chunk]:
        """批量创建分块

        Args:
            chunks: 分块数据列表
            kb_id: 知识库 ID
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            创建的分块列表
        """
        chunk_objs = [
            Chunk(
                id=str(uuid4()),
                knowledge_base_id=kb_id,
                knowledge_id=knowledge_id,
                tenant_id=tenant_id,
                **chunk,
            )
            for chunk in chunks
        ]
        for chunk in chunk_objs:
            self.session.add(chunk)
        await self.session.commit()
        return chunk_objs

    async def delete_by_knowledge(self, knowledge_id: str, tenant_id: int) -> bool:
        """删除知识条目的所有分块

        Args:
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        from datetime import UTC, datetime

        stmt = select(Chunk).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        for chunk in chunks:
            chunk.deleted_at = datetime.now(UTC)

        await self.session.commit()
        return True


__all__ = [
    "KnowledgeBaseRepository",
    "KnowledgeRepository",
    "ChunkRepository",
]
