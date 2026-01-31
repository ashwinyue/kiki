"""文档分块仓储

对齐 WeKnora99 分块管理
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import Chunk
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginationParams, PaginatedResult

logger = get_logger(__name__)


class ChunkRepository(BaseRepository[Chunk]):
    """文档分块仓储"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        """通过 ID 获取分块"""
        stmt = select(Chunk).where(Chunk.id == chunk_id, Chunk.deleted_at.is_(None))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_tenant(self, chunk_id: str, tenant_id: int) -> Chunk | None:
        """通过 ID 和租户获取分块"""
        stmt = select(Chunk).where(
            Chunk.id == chunk_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_knowledge(
        self,
        knowledge_id: str,
        tenant_id: int,
        params: PaginationParams,
        chunk_types: list[str] | None = None,
    ) -> PaginatedResult[Chunk]:
        """获取知识的分块列表（带分页）"""
        # 构建基础查询
        stmt = select(Chunk).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )

        # 添加分块类型过滤
        if chunk_types:
            stmt = stmt.where(Chunk.chunk_type.in_(chunk_types))

        # 获取总数
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0

        # 添加分页和排序
        stmt = stmt.order_by(Chunk.chunk_index).offset((params.page - 1) * params.size).limit(params.size)

        # 执行查询
        result = await self.session.execute(stmt)
        items = result.scalars().all()

        return PaginatedResult(
            items=list(items),
            total=total,
            page=params.page,
            size=params.size,
        )

    async def update(self, chunk: Chunk) -> Chunk:
        """更新分块"""
        self.session.add(chunk)
        await self.session.commit()
        await self.session.refresh(chunk)
        return chunk

    async def update_fields(
        self,
        chunk_id: str,
        tenant_id: int,
        **fields: Any,
    ) -> Chunk | None:
        """更新分块字段"""
        stmt = (
            update(Chunk)
            .where(Chunk.id == chunk_id, Chunk.tenant_id == tenant_id, Chunk.deleted_at.is_(None))
            .values(**fields)
            .returning(Chunk)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def soft_delete(self, chunk_id: str, tenant_id: int) -> bool:
        """软删除分块"""
        from datetime import datetime, UTC

        stmt = (
            update(Chunk)
            .where(Chunk.id == chunk_id, Chunk.tenant_id == tenant_id)
            .values(deleted_at=datetime.now(UTC))
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def delete_by_knowledge(self, knowledge_id: str, tenant_id: int) -> int:
        """删除知识下的所有分块（硬删除）"""
        stmt = delete(Chunk).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.tenant_id == tenant_id,
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def delete_by_knowledge_base(
        self, knowledge_base_id: str, tenant_id: int
    ) -> int:
        """删除知识库下的所有分块（硬删除）

        Args:
            knowledge_base_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            删除的记录数
        """
        stmt = delete(Chunk).where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Chunk.tenant_id == tenant_id,
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

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
            knowledge_id: 知识 ID
            tenant_id: 租户 ID

        Returns:
            创建的分块列表
        """
        import uuid
        from datetime import UTC

        now = datetime.now(UTC)
        new_chunks = []

        for chunk_data in chunks:
            chunk = Chunk(
                id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                knowledge_base_id=kb_id,
                tenant_id=tenant_id,
                content=chunk_data.get("content", ""),
                chunk_index=chunk_data.get("chunk_index", 0),
                chunk_type=chunk_data.get("chunk_type", "text"),
                start_at=chunk_data.get("start_at", 0),
                end_at=chunk_data.get("end_at", 0),
                is_enabled=chunk_data.get("is_enabled", True),
                embedding=chunk_data.get("embedding"),
                meta_data=chunk_data.get("meta_data", {}),
                created_at=now,
                updated_at=now,
            )
            self.session.add(chunk)
            new_chunks.append(chunk)

        await self.session.commit()

        # 刷新获取 ID
        for chunk in new_chunks:
            await self.session.refresh(chunk)

        logger.info(
            "chunks_created",
            knowledge_id=knowledge_id,
            count=len(new_chunks),
        )

        return new_chunks

    async def count_by_knowledge(self, knowledge_id: str, tenant_id: int) -> int:
        """统计知识的分块数量"""
        stmt = select(func.count()).where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def delete_generated_question(
        self,
        chunk_id: str,
        question_id: str,
        tenant_id: int,
    ) -> bool:
        """从分块元数据中删除生成的问题

        生成的问题存储在 metadata.generated_questions 数组中
        """
        chunk = await self.get_by_tenant(chunk_id, tenant_id)
        if not chunk:
            return False

        # 获取当前元数据
        metadata = chunk.metadata or {}
        generated_questions = metadata.get("generated_questions", [])

        # 移除指定问题
        updated_questions = [q for q in generated_questions if q.get("id") != question_id]

        # 检查是否找到并删除
        if len(updated_questions) == len(generated_questions):
            return False

        # 更新元数据
        metadata["generated_questions"] = updated_questions
        chunk.metadata = metadata
        await self.session.commit()

        logger.info(
            "generated_question_deleted",
            chunk_id=chunk_id,
            question_id=question_id,
        )
        return True


__all__ = ["ChunkRepository", "PaginatedResult"]
