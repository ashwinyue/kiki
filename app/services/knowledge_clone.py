"""知识库克隆服务

提供知识库深度复制功能，包括配置、文档、向量、标签等。
对齐 WeKnora99 的 Knowledge Base Copy API。
"""

import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.redis import RedisCache
from app.models.knowledge import Chunk, Knowledge, KnowledgeBase
from app.observability.logging import get_logger
from app.repositories.knowledge import (
    ChunkRepository,
    KnowledgeBaseRepository,
    KnowledgeRepository,
)

logger = get_logger(__name__)

# Redis 缓存前缀
COPY_PROGRESS_PREFIX = "kiki:kb_copy:progress:"


class CopyTaskStatus:
    """复制任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CopyProgress:
    """复制进度信息"""

    def __init__(
        self,
        task_id: str,
        source_id: str,
        target_id: str,
        tenant_id: int,
        status: str = CopyTaskStatus.PENDING,
    ) -> None:
        self.task_id = task_id
        self.source_id = source_id
        self.target_id = target_id
        self.tenant_id = tenant_id
        self.status = status
        self.message = "任务已创建"
        self.total_knowledges = 0
        self.copied_knowledges = 0
        self.total_chunks = 0
        self.copied_chunks = 0
        self.total_tags = 0
        self.copied_tags = 0
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "message": self.message,
            "total_knowledges": self.total_knowledges,
            "copied_knowledges": self.copied_knowledges,
            "total_chunks": self.total_chunks,
            "copied_chunks": self.copied_chunks,
            "total_tags": self.total_tags,
            "copied_tags": self.copied_tags,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "progress_percent": self._calculate_percent(),
        }

    def _calculate_percent(self) -> float:
        """计算总进度百分比"""
        total = self.total_knowledges + self.total_chunks + self.total_tags
        copied = self.copied_knowledges + self.copied_chunks + self.copied_tags
        if total == 0:
            return 0.0
        return round((copied / total) * 100, 2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CopyProgress":
        """从字典创建实例"""
        progress = cls(
            task_id=data["task_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            tenant_id=data["tenant_id"],
            status=data["status"],
        )
        progress.message = data.get("message", "任务已创建")
        progress.total_knowledges = data.get("total_knowledges", 0)
        progress.copied_knowledges = data.get("copied_knowledges", 0)
        progress.total_chunks = data.get("total_chunks", 0)
        progress.copied_chunks = data.get("copied_chunks", 0)
        progress.total_tags = data.get("total_tags", 0)
        progress.copied_tags = data.get("copied_tags", 0)
        progress.error = data.get("error")
        if started_at := data.get("started_at"):
            progress.started_at = datetime.fromisoformat(started_at)
        if completed_at := data.get("completed_at"):
            progress.completed_at = datetime.fromisoformat(completed_at)
        return progress


class KnowledgeCloner:
    """知识库克隆器

    提供知识库的深度复制功能，支持：
    - 基础配置复制
    - 文档知识条目复制
    - URL 知识条目复制
    - 手工知识条目复制
    - 文档分块复制
    - 标签复制
    - 异步后台任务
    - 进度跟踪
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化知识库克隆器

        Args:
            session: 异步数据库会话
        """
        self.session = session
        self.kb_repo = KnowledgeBaseRepository(session)
        self.knowledge_repo = KnowledgeRepository(session)
        self.chunk_repo = ChunkRepository(session)
        self.cache = RedisCache(key_prefix=COPY_PROGRESS_PREFIX)

    async def start_copy_task(
        self,
        source_id: str,
        target_id: str | None,
        tenant_id: int,
        target_name: str | None = None,
    ) -> str:
        """启动知识库复制任务

        Args:
            source_id: 源知识库 ID
            target_id: 目标知识库 ID（为空时创建新知识库）
            tenant_id: 租户 ID
            target_name: 目标知识库名称（仅当创建新知识库时使用）

        Returns:
            任务 ID
        """
        # 生成任务 ID
        task_id = str(uuid.uuid4())

        # 验证源知识库
        source_kb = await self.kb_repo.get_by_tenant(source_id, tenant_id)
        if not source_kb:
            raise ValueError(f"源知识库 {source_id} 不存在")

        # 确定目标知识库
        if target_id:
            target_kb = await self.kb_repo.get_by_tenant(target_id, tenant_id)
            if not target_kb:
                raise ValueError(f"目标知识库 {target_id} 不存在")
        else:
            # 创建新知识库
            target_kb = await self._create_target_kb(
                source_kb, tenant_id, target_name
            )
            target_id = target_kb.id

        # 初始化进度
        progress = CopyProgress(
            task_id=task_id,
            source_id=source_id,
            target_id=target_id,
            tenant_id=tenant_id,
        )
        await self._save_progress(progress)

        # 统计总数
        progress.total_knowledges = await self._count_knowledges(source_id, tenant_id)
        progress.total_chunks = await self._count_chunks(source_id, tenant_id)
        progress.total_tags = await self._count_tags(source_id, tenant_id)
        progress.started_at = datetime.now(UTC)
        progress.status = CopyTaskStatus.RUNNING
        progress.message = "复制任务已启动"
        await self._save_progress(progress)

        logger.info(
            "knowledge_copy_task_started",
            task_id=task_id,
            source_id=source_id,
            target_id=target_id,
            tenant_id=tenant_id,
            total_knowledges=progress.total_knowledges,
            total_chunks=progress.total_chunks,
            total_tags=progress.total_tags,
        )

        return task_id

    async def get_progress(self, task_id: str) -> CopyProgress | None:
        """获取复制进度

        Args:
            task_id: 任务 ID

        Returns:
            复制进度信息
        """
        data = await self.cache.get(task_id)
        if not data:
            return None
        try:
            return CopyProgress.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("failed_to_parse_progress", task_id=task_id, error=str(e))
            return None

    async def execute_copy(
        self,
        task_id: str,
    ) -> CopyProgress:
        """执行知识库复制

        Args:
            task_id: 任务 ID

        Returns:
            最终复制进度
        """
        progress = await self.get_progress(task_id)
        if not progress:
            raise ValueError(f"任务 {task_id} 不存在")

        if progress.status == CopyTaskStatus.COMPLETED:
            return progress

        try:
            progress.status = CopyTaskStatus.RUNNING
            progress.message = "正在复制知识库配置..."
            await self._save_progress(progress)

            # 1. 复制标签
            await self._copy_tags(progress)

            # 2. 复制知识条目和分块
            await self._copy_knowledges(progress)

            # 3. 完成
            progress.status = CopyTaskStatus.COMPLETED
            progress.message = "复制完成"
            progress.completed_at = datetime.now(UTC)

            logger.info(
                "knowledge_copy_completed",
                task_id=task_id,
                source_id=progress.source_id,
                target_id=progress.target_id,
                copied_knowledges=progress.copied_knowledges,
                copied_chunks=progress.copied_chunks,
                copied_tags=progress.copied_tags,
            )

        except Exception as e:
            progress.status = CopyTaskStatus.FAILED
            progress.error = str(e)
            progress.message = f"复制失败: {str(e)}"
            progress.completed_at = datetime.now(UTC)

            logger.error(
                "knowledge_copy_failed",
                task_id=task_id,
                source_id=progress.source_id,
                target_id=progress.target_id,
                error=str(e),
            )

        await self._save_progress(progress)
        return progress

    async def _create_target_kb(
        self,
        source_kb: KnowledgeBase,
        tenant_id: int,
        name: str | None = None,
    ) -> KnowledgeBase:
        """创建目标知识库

        Args:
            source_kb: 源知识库
            tenant_id: 租户 ID
            name: 目标知识库名称

        Returns:
            新创建的知识库
        """
        # 深度复制配置
        chunking_config = deepcopy(source_kb.chunking_config or {})
        image_processing_config = deepcopy(source_kb.image_processing_config or {})
        cos_config = deepcopy(source_kb.cos_config or {})
        vlm_config = deepcopy(source_kb.vlm_config or {})

        create_data = {
            "name": name or f"{source_kb.name} (Copy)",
            "description": source_kb.description,
            "kb_type": source_kb.kb_type,
            "chunking_config": chunking_config,
            "embedding_model_id": source_kb.embedding_model_id,
            "summary_model_id": source_kb.summary_model_id,
            "rerank_model_id": source_kb.rerank_model_id,
            "image_processing_config": image_processing_config,
            "cos_config": cos_config,
            "vlm_config": vlm_config,
            "is_temporary": False,
        }

        return await self.kb_repo.create_with_tenant(create_data, tenant_id)

    async def _count_knowledges(self, kb_id: str, tenant_id: int) -> int:
        """统计知识条目数量"""
        stmt = select(func.count()).select_from(Knowledge).where(
            Knowledge.knowledge_base_id == kb_id,
            Knowledge.tenant_id == tenant_id,
            Knowledge.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def _count_chunks(self, kb_id: str, tenant_id: int) -> int:
        """统计分块数量"""
        stmt = select(func.count()).select_from(Chunk).where(
            Chunk.knowledge_base_id == kb_id,
            Chunk.tenant_id == tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def _count_tags(self, kb_id: str, tenant_id: int) -> int:
        """统计标签数量"""
        from app.models.knowledge import KnowledgeTag

        stmt = select(func.count()).select_from(KnowledgeTag).where(
            KnowledgeTag.knowledge_base_id == kb_id,
            KnowledgeTag.tenant_id == tenant_id,
            KnowledgeTag.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def _copy_tags(self, progress: CopyProgress) -> None:
        """复制标签

        Args:
            progress: 复制进度
        """
        from app.models.knowledge import KnowledgeTag

        progress.message = "正在复制标签..."
        await self._save_progress(progress)

        # 查询源标签
        stmt = select(KnowledgeTag).where(
            KnowledgeTag.knowledge_base_id == progress.source_id,
            KnowledgeTag.tenant_id == progress.tenant_id,
            KnowledgeTag.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        source_tags = result.scalars().all()

        # 复制标签
        for tag in source_tags:
            new_tag = KnowledgeTag(
                id=str(uuid.uuid4()),
                knowledge_base_id=progress.target_id,
                tenant_id=progress.tenant_id,
                name=tag.name,
                color=tag.color,
                sort_order=tag.sort_order,
            )
            self.session.add(new_tag)
            progress.copied_tags += 1

        await self.session.commit()
        await self._save_progress(progress)

        logger.info(
            "tags_copied",
            task_id=progress.task_id,
            count=progress.copied_tags,
        )

    async def _copy_knowledges(self, progress: CopyProgress) -> None:
        """复制知识条目和分块

        Args:
            progress: 复制进度
        """
        progress.message = "正在复制知识条目..."
        await self._save_progress(progress)

        # 分批查询知识条目
        batch_size = 100
        offset = 0

        while True:
            stmt = select(Knowledge).where(
                Knowledge.knowledge_base_id == progress.source_id,
                Knowledge.tenant_id == progress.tenant_id,
                Knowledge.deleted_at.is_(None),
            ).offset(offset).limit(batch_size)

            result = await self.session.execute(stmt)
            knowledges = result.scalars().all()

            if not knowledges:
                break

            for knowledge in knowledges:
                await self._copy_knowledge(knowledge, progress)

            offset += batch_size
            await self._save_progress(progress)

        logger.info(
            "knowledges_copied",
            task_id=progress.task_id,
            count=progress.copied_knowledges,
            chunks=progress.copied_chunks,
        )

    async def _copy_knowledge(
        self,
        source_knowledge: Knowledge,
        progress: CopyProgress,
    ) -> None:
        """复制单个知识条目及其分块

        Args:
            source_knowledge: 源知识条目
            progress: 复制进度
        """
        # 创建新知识条目
        new_knowledge = Knowledge(
            id=str(uuid.uuid4()),
            knowledge_base_id=progress.target_id,
            tenant_id=progress.tenant_id,
            type=source_knowledge.type,
            title=source_knowledge.title,
            description=source_knowledge.description,
            source=source_knowledge.source,
            parse_status=source_knowledge.parse_status,
            enable_status=source_knowledge.enable_status,
            embedding_model_id=source_knowledge.embedding_model_id,
            summary_status=source_knowledge.summary_status,
            file_name=source_knowledge.file_name,
            file_type=source_knowledge.file_type,
            file_size=source_knowledge.file_size,
            file_path=source_knowledge.file_path,
            file_hash=source_knowledge.file_hash,
            storage_size=source_knowledge.storage_size,
            meta_data=deepcopy(source_knowledge.meta_data)
            if source_knowledge.meta_data
            else None,
        )
        self.session.add(new_knowledge)
        await self.session.flush()  # 获取 ID

        progress.copied_knowledges += 1

        # 复制分块
        await self._copy_chunks(source_knowledge.id, new_knowledge.id, progress)

    async def _copy_chunks(
        self,
        source_knowledge_id: str,
        target_knowledge_id: str,
        progress: CopyProgress,
    ) -> None:
        """复制知识条目的分块

        Args:
            source_knowledge_id: 源知识条目 ID
            target_knowledge_id: 目标知识条目 ID
            progress: 复制进度
        """
        stmt = select(Chunk).where(
            Chunk.knowledge_id == source_knowledge_id,
            Chunk.tenant_id == progress.tenant_id,
            Chunk.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        chunk_id_mapping: dict[str, str] = {}

        for chunk in chunks:
            new_chunk = Chunk(
                id=str(uuid.uuid4()),
                knowledge_base_id=progress.target_id,
                knowledge_id=target_knowledge_id,
                tenant_id=progress.tenant_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                is_enabled=chunk.is_enabled,
                start_at=chunk.start_at,
                end_at=chunk.end_at,
                chunk_type=chunk.chunk_type,
                tag_id=chunk.tag_id,
                image_info=chunk.image_info,
                meta_data=deepcopy(chunk.meta_data) if chunk.meta_data else None,
                content_hash=chunk.content_hash,
                flags=chunk.flags,
                status=chunk.status,
            )
            self.session.add(new_chunk)

            # 保存 ID 映射，用于更新关联
            chunk_id_mapping[chunk.id] = new_chunk.id
            progress.copied_chunks += 1

        await self.session.flush()

        # 更新分块关联（pre_chunk_id, next_chunk_id, parent_chunk_id）
        for chunk in chunks:
            new_id = chunk_id_mapping[chunk.id]
            stmt = select(Chunk).where(Chunk.id == new_id)
            result = await self.session.execute(stmt)
            new_chunk = result.scalar_one_or_none()

            if new_chunk:
                if chunk.pre_chunk_id and chunk.pre_chunk_id in chunk_id_mapping:
                    new_chunk.pre_chunk_id = chunk_id_mapping[chunk.pre_chunk_id]
                if chunk.next_chunk_id and chunk.next_chunk_id in chunk_id_mapping:
                    new_chunk.next_chunk_id = chunk_id_mapping[chunk.next_chunk_id]
                if (
                    chunk.parent_chunk_id
                    and chunk.parent_chunk_id in chunk_id_mapping
                ):
                    new_chunk.parent_chunk_id = chunk_id_mapping[chunk.parent_chunk_id]

        await self.session.commit()

    async def _save_progress(self, progress: CopyProgress) -> None:
        """保存进度到 Redis

        Args:
            progress: 复制进度
        """
        data = json.dumps(progress.to_dict())
        # 设置 24 小时过期
        await self.cache.set(progress.task_id, data, ttl=86400)


# 便捷函数


async def create_copy_task(
    session: AsyncSession,
    source_id: str,
    target_id: str | None,
    tenant_id: int,
    target_name: str | None = None,
) -> tuple[str, CopyProgress]:
    """创建知识库复制任务（便捷函数）

    Args:
        session: 数据库会话
        source_id: 源知识库 ID
        target_id: 目标知识库 ID
        tenant_id: 租户 ID
        target_name: 目标知识库名称

    Returns:
        (任务 ID, 初始进度)
    """
    cloner = KnowledgeCloner(session)
    task_id = await cloner.start_copy_task(
        source_id, target_id, tenant_id, target_name
    )
    progress = await cloner.get_progress(task_id)
    return task_id, progress


async def get_copy_progress(
    session: AsyncSession,
    task_id: str,
) -> CopyProgress | None:
    """获取复制进度（便捷函数）

    Args:
        session: 数据库会话
        task_id: 任务 ID

    Returns:
        复制进度
    """
    cloner = KnowledgeCloner(session)
    return await cloner.get_progress(task_id)


async def execute_copy_task(
    session: AsyncSession,
    task_id: str,
) -> CopyProgress:
    """执行复制任务（便捷函数）

    Args:
        session: 数据库会话
        task_id: 任务 ID

    Returns:
        最终进度
    """
    cloner = KnowledgeCloner(session)
    return await cloner.execute_copy(task_id)


# 导入 func（延迟导入避免循环）
from sqlalchemy import func  # noqa: E402

__all__ = [
    "KnowledgeCloner",
    "CopyProgress",
    "CopyTaskStatus",
    "create_copy_task",
    "get_copy_progress",
    "execute_copy_task",
]
