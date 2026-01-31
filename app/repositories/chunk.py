"""文档分块仓储

对齐 WeKnora99 分块管理
"""

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
