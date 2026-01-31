"""FAQ 数据仓储

提供 FAQ 的数据访问操作。
"""

from datetime import UTC, datetime

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.faq import FAQ, FAQCategory, FAQCreate, FAQStatus, FAQUpdate
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginationParams

logger = get_logger(__name__)


class FAQRepository(BaseRepository[FAQ]):
    """FAQ 仓储

    提供 FAQ 的 CRUD 操作和特定查询方法。
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(FAQ, session)

    # ============== 基础 CRUD ==============

    async def create(
        self,
        data: FAQCreate,
        tenant_id: int | None = None,
        created_by: int | None = None,
    ) -> FAQ:
        """创建 FAQ

        Args:
            data: 创建数据
            tenant_id: 租户 ID
            created_by: 创建人 ID

        Returns:
            创建的 FAQ
        """
        faq = FAQ(
            question=data.question,
            answer=data.answer,
            category=data.category,
            tags=data.tags,
            priority=data.priority,
            locale=data.locale,
            slug=data.slug,
            tenant_id=tenant_id,
            created_by=created_by,
            updated_by=created_by,
        )

        self.session.add(faq)
        await self.session.commit()
        await self.session.refresh(faq)

        logger.info(
            "faq_created",
            faq_id=faq.id,
            tenant_id=tenant_id,
            category=data.category.value,
        )

        return faq

    async def update_faq(
        self,
        faq: FAQ,
        data: FAQUpdate,
        updated_by: int | None = None,
    ) -> FAQ:
        """更新 FAQ

        Args:
            faq: FAQ 实例
            data: 更新数据
            updated_by: 更新人 ID

        Returns:
            更新后的 FAQ
        """
        if data.question is not None:
            faq.question = data.question
        if data.answer is not None:
            faq.answer = data.answer
        if data.category is not None:
            faq.category = data.category
        if data.tags is not None:
            faq.tags = data.tags
        if data.priority is not None:
            faq.priority = data.priority
        if data.status is not None:
            faq.status = data.status
            # 如果状态从未发布变为已发布，设置发布时间
            if data.status == FAQStatus.PUBLISHED and not faq.published_at:
                faq.published_at = datetime.now(UTC)
        if data.locale is not None:
            faq.locale = data.locale
        if data.slug is not None:
            faq.slug = data.slug

        faq.updated_by = updated_by
        faq.updated_at = datetime.now(UTC)

        await self.session.commit()
        await self.session.refresh(faq)

        logger.info("faq_updated", faq_id=faq.id)

        return faq

    # ============== 查询方法 ==============

    async def list_by_status(
        self,
        status: FAQStatus,
        *,
        offset: int = 0,
        limit: int = 100,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """按状态列出 FAQ

        Args:
            status: 状态
            offset: 偏移量
            limit: 限制数量
            tenant_id: 租户 ID（可选）

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(FAQ.status == status)

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.order_by(desc(FAQ.priority), FAQ.created_at)
        statement = statement.offset(offset).limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_category(
        self,
        category: FAQCategory,
        *,
        offset: int = 0,
        limit: int = 100,
        status: FAQStatus | None = None,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """按分类列出 FAQ

        Args:
            category: 分类
            offset: 偏移量
            limit: 限制数量
            status: 状态筛选
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(FAQ.category == category)

        if status:
            statement = statement.where(FAQ.status == status)

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.order_by(desc(FAQ.priority), FAQ.created_at)
        statement = statement.offset(offset).limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_locale(
        self,
        locale: str,
        *,
        offset: int = 0,
        limit: int = 100,
        status: FAQStatus | None = None,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """按语言列出 FAQ

        Args:
            locale: 语言代码
            offset: 偏移量
            limit: 限制数量
            status: 状态筛选
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(FAQ.locale == locale)

        if status:
            statement = statement.where(FAQ.status == status)

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.order_by(desc(FAQ.priority), FAQ.created_at)
        statement = statement.offset(offset).limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def list_by_tags(
        self,
        tags: list[str],
        *,
        offset: int = 0,
        limit: int = 100,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """按标签列出 FAQ

        Args:
            tags: 标签列表（满足任一标签即可）
            offset: 偏移量
            limit: 限制数量
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(FAQ.tags.overlap(tags))

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.order_by(desc(FAQ.priority), FAQ.created_at)
        statement = statement.offset(offset).limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_by_slug(self, slug: str) -> FAQ | None:
        """根据 slug 获取 FAQ

        Args:
            slug: URL 标识符

        Returns:
            FAQ 对象
        """
        statement = select(FAQ).where(FAQ.slug == slug)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        locale: str | None = None,
        category: FAQCategory | None = None,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """搜索 FAQ

        在问题和答案中搜索关键词。

        Args:
            query: 搜索关键词
            limit: 返回数量限制
            locale: 语言筛选
            category: 分类筛选
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(
            or_(
                FAQ.question.ilike(f"%{query}%"),
                FAQ.answer.ilike(f"%{query}%"),
            )
        )

        if locale:
            statement = statement.where(FAQ.locale == locale)

        if category:
            statement = statement.where(FAQ.category == category)

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        # 优先显示已发布的，按优先级排序
        statement = statement.order_by(
            desc(FAQ.status == FAQStatus.PUBLISHED),
            desc(FAQ.priority),
            FAQ.created_at,
        )
        statement = statement.limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    # ============== 统计方法 ==============

    async def count_by_status(
        self,
        tenant_id: int | None = None,
    ) -> dict[str, int]:
        """按状态统计 FAQ 数量

        Args:
            tenant_id: 租户 ID

        Returns:
            状态统计字典
        """
        statement = select(FAQ.status, func.count(FAQ.id))

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.group_by(FAQ.status)

        result = await self.session.execute(statement)
        return {status.value: count for status, count in result.all()}

    async def count_by_category(
        self,
        tenant_id: int | None = None,
    ) -> dict[str, int]:
        """按分类统计 FAQ 数量

        Args:
            tenant_id: 租户 ID

        Returns:
            分类统计字典
        """
        statement = select(FAQ.category, func.count(FAQ.id))

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.group_by(FAQ.category)

        result = await self.session.execute(statement)
        return {category.value: count for category, count in result.all()}

    async def get_most_viewed(
        self,
        limit: int = 5,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """获取浏览最多的 FAQ

        Args:
            limit: 返回数量
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        statement = select(FAQ).where(FAQ.view_count > 0)

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        statement = statement.order_by(desc(FAQ.view_count)).limit(limit)

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_most_helpful(
        self,
        limit: int = 5,
        tenant_id: int | None = None,
    ) -> list[FAQ]:
        """获取最有用的 FAQ（按有用/无用比例）

        Args:
            limit: 返回数量
            tenant_id: 租户 ID

        Returns:
            FAQ 列表
        """
        # 计算有用率，只统计有反馈的
        total_feedback = FAQ.helpful_count + FAQ.not_helpful_count
        statement = (
            select(FAQ)
            .where(
                and_(
                    total_feedback > 0,
                    FAQ.status == FAQStatus.PUBLISHED,
                )
            )
            .order_by(
                desc(
                    (FAQ.helpful_count.cast(float) / total_feedback.cast(float))
                ),
                desc(FAQ.helpful_count),
            )
            .limit(limit)
        )

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    # ============== 反馈方法 ==============

    async def increment_view_count(self, faq_id: int) -> bool:
        """增加浏览次数

        Args:
            faq_id: FAQ ID

        Returns:
            是否成功
        """
        faq = await self.get(faq_id)
        if faq:
            faq.view_count += 1
            await self.session.commit()
            return True
        return False

    async def add_feedback(
        self,
        faq_id: int,
        helpful: bool,
    ) -> bool:
        """添加反馈

        Args:
            faq_id: FAQ ID
            helpful: 是否有用

        Returns:
            是否成功
        """
        faq = await self.get(faq_id)
        if faq:
            if helpful:
                faq.helpful_count += 1
            else:
                faq.not_helpful_count += 1
            await self.session.commit()
            logger.info(
                "faq_feedback_added",
                faq_id=faq_id,
                helpful=helpful,
            )
            return True
        return False

    # ============== 分页查询 ==============

    async def list_paginated(
        self,
        params: PaginationParams,
        *,
        status: FAQStatus | None = None,
        category: FAQCategory | None = None,
        locale: str | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
        tenant_id: int | None = None,
    ) -> tuple[list[FAQ], int]:
        """分页查询 FAQ

        Args:
            params: 分页参数
            status: 状态筛选
            category: 分类筛选
            locale: 语言筛选
            tags: 标签筛选
            search: 搜索关键词
            tenant_id: 租户 ID

        Returns:
            (FAQ 列表, 总数)
        """
        statement = select(FAQ)

        # 应用筛选条件
        if status:
            statement = statement.where(FAQ.status == status)

        if category:
            statement = statement.where(FAQ.category == category)

        if locale:
            statement = statement.where(FAQ.locale == locale)

        if tags:
            statement = statement.where(FAQ.tags.overlap(tags))

        if search:
            statement = statement.where(
                or_(
                    FAQ.question.ilike(f"%{search}%"),
                    FAQ.answer.ilike(f"%{search}%"),
                )
            )

        if tenant_id is not None:
            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        # 获取总数
        count_statement = select(func.count()).select_from(statement.subquery())
        total_result = await self.session.execute(count_statement)
        total = total_result.scalar() or 0

        # 获取分页数据
        statement = statement.order_by(desc(FAQ.priority), FAQ.created_at.desc())
        statement = statement.offset(params.offset).limit(params.limit)

        items_result = await self.session.execute(statement)
        items = list(items_result.scalars().all())

        return items, total
