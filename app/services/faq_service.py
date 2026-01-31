"""FAQ 服务层

提供 FAQ 的 CRUD 业务逻辑，包括创建、查询、更新、删除、搜索等。
封装响应构建和权限检查逻辑，消除路由层的重复代码。
"""

import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.models.faq import (
    FAQ,
    FAQCategory,
    FAQCreate,
    FAQDetail,
    FAQRead,
    FAQSearchResult,
    FAQStatus,
    FAQUpdate,
)
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.faq import FAQRepository
from app.schemas.faq import (
    FAQBulkUpdateResponse,
    FAQFeedbackResponse,
    FAQListResponse,
    FAQSearchResponse,
    FAQStatsResponse,
)

logger = get_logger(__name__)


class FAQService:
    """FAQ 管理服务

    封装 FAQ 的业务逻辑，包括：
    - 创建和更新 FAQ
    - 查询和列表
    - 搜索功能
    - 反馈统计
    - 权限验证
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化服务

        Args:
            session: 数据库会话
        """
        self.session = session
        self._repository: FAQRepository | None = None

    @property
    def repository(self) -> FAQRepository:
        """获取仓储（延迟初始化）"""
        if self._repository is None:
            self._repository = FAQRepository(self.session)
        return self._repository

    # ============== 响应构建方法 ==============

    def _to_read(self, faq: FAQ) -> FAQRead:
        """转换为 FAQ 读取模型

        Args:
            faq: FAQ 模型

        Returns:
            FAQRead 读取模型
        """
        return FAQRead(
            id=faq.id,
            question=faq.question,
            answer=faq.answer,
            category=faq.category,
            tags=faq.tags,
            priority=faq.priority,
            locale=faq.locale,
            status=faq.status,
            slug=faq.slug,
            view_count=faq.view_count,
            helpful_count=faq.helpful_count,
            not_helpful_count=faq.not_helpful_count,
            created_at=faq.created_at,
            updated_at=faq.updated_at,
            published_at=faq.published_at,
        )

    def _to_detail(self, faq: FAQ) -> FAQDetail:
        """转换为 FAQ 详情模型

        Args:
            faq: FAQ 模型

        Returns:
            FAQDetail 详情模型
        """
        return FAQDetail(
            id=faq.id,
            question=faq.question,
            answer=faq.answer,
            category=faq.category,
            tags=faq.tags,
            priority=faq.priority,
            locale=faq.locale,
            status=faq.status,
            slug=faq.slug,
            view_count=faq.view_count,
            helpful_count=faq.helpful_count,
            not_helpful_count=faq.not_helpful_count,
            created_at=faq.created_at,
            updated_at=faq.updated_at,
            published_at=faq.published_at,
            tenant_id=faq.tenant_id,
            created_by=faq.created_by,
            updated_by=faq.updated_by,
        )

    def _to_search_result(
        self,
        faq: FAQ,
        relevance_score: float | None = None,
        rank: int | None = None,
    ) -> FAQSearchResult:
        """转换为搜索结果模型

        Args:
            faq: FAQ 模型
            relevance_score: 相关性评分
            rank: 排名

        Returns:
            FAQSearchResult 搜索结果模型
        """
        return FAQSearchResult(
            id=faq.id,
            question=faq.question,
            answer=faq.answer,
            category=faq.category,
            tags=faq.tags,
            locale=faq.locale,
            slug=faq.slug,
            relevance_score=relevance_score,
            rank=rank,
        )

    # ============== CRUD 操作 ==============

    async def create_faq(
        self,
        data: FAQCreate,
        user_id: int,
        tenant_id: int | None = None,
    ) -> FAQRead:
        """创建 FAQ

        Args:
            data: 创建数据
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            创建的 FAQRead
        """
        # 检查 slug 是否已存在
        if data.slug:
            existing = await self.repository.get_by_slug(data.slug)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"FAQ with slug '{data.slug}' already exists",
                )

        faq = await self.repository.create(
            data=data,
            tenant_id=tenant_id,
            created_by=user_id,
        )

        logger.info(
            "faq_created",
            faq_id=faq.id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        return self._to_read(faq)

    async def list_faqs(
        self,
        *,
        page: int = 1,
        size: int = 20,
        status: FAQStatus | None = None,
        category: FAQCategory | None = None,
        locale: str | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
        tenant_id: int | None = None,
    ) -> FAQListResponse:
        """列出 FAQ

        Args:
            page: 页码
            size: 每页数量
            status: 状态筛选
            category: 分类筛选
            locale: 语言筛选
            tags: 标签筛选
            search: 搜索关键词
            tenant_id: 租户 ID

        Returns:
            FAQListResponse 列表响应
        """
        params = PaginationParams(page=page, size=size)
        items, total = await self.repository.list_paginated(
            params=params,
            status=status,
            category=category,
            locale=locale,
            tags=tags,
            search=search,
            tenant_id=tenant_id,
        )

        # 计算总页数
        pages = (total + size - 1) // size if size > 0 else 0

        return FAQListResponse(
            items=[self._to_read(item) for item in items],
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

    async def get_faq(
        self,
        faq_id: int,
        *,
        increment_view: bool = True,
    ) -> FAQDetail:
        """获取 FAQ 详情

        Args:
            faq_id: FAQ ID
            increment_view: 是否增加浏览次数

        Returns:
            FAQDetail 详情模型

        Raises:
            HTTPException: FAQ 不存在时返回 404
        """
        faq = await self.repository.get(faq_id)

        if not faq:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ not found",
            )

        # 增加浏览次数
        if increment_view:
            await self.repository.increment_view_count(faq_id)

        return self._to_detail(faq)

    async def get_faq_by_slug(self, slug: str) -> FAQDetail:
        """根据 slug 获取 FAQ 详情

        Args:
            slug: URL 标识符

        Returns:
            FAQDetail 详情模型

        Raises:
            HTTPException: FAQ 不存在时返回 404
        """
        faq = await self.repository.get_by_slug(slug)

        if not faq:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ not found",
            )

        # 增加浏览次数
        await self.repository.increment_view_count(faq.id)

        return self._to_detail(faq)

    async def update_faq(
        self,
        faq_id: int,
        data: FAQUpdate,
        user_id: int,
    ) -> FAQRead:
        """更新 FAQ

        Args:
            faq_id: FAQ ID
            data: 更新数据
            user_id: 用户 ID

        Returns:
            更新后的 FAQRead

        Raises:
            HTTPException: FAQ 不存在时返回 404
        """
        faq = await self.repository.get(faq_id)

        if not faq:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ not found",
            )

        # 检查 slug 是否冲突
        if data.slug and data.slug != faq.slug:
            existing = await self.repository.get_by_slug(data.slug)
            if existing and existing.id != faq_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"FAQ with slug '{data.slug}' already exists",
                )

        updated = await self.repository.update_faq(faq, data, updated_by=user_id)

        logger.info(
            "faq_updated",
            faq_id=faq_id,
            user_id=user_id,
        )

        return self._to_read(updated)

    async def delete_faq(self, faq_id: int) -> None:
        """删除 FAQ

        Args:
            faq_id: FAQ ID

        Raises:
            HTTPException: FAQ 不存在或删除失败时
        """
        success = await self.repository.delete(faq_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ not found or delete failed",
            )

        logger.info("faq_deleted", faq_id=faq_id)

    # ============== 搜索功能 ==============

    async def search_faqs(
        self,
        query: str,
        *,
        locale: str = "zh-CN",
        category: FAQCategory | None = None,
        limit: int = 10,
        tenant_id: int | None = None,
    ) -> FAQSearchResponse:
        """搜索 FAQ

        Args:
            query: 搜索查询
            locale: 语言
            category: 分类
            limit: 返回数量限制
            tenant_id: 租户 ID

        Returns:
            FAQSearchResponse 搜索响应
        """
        start_time = time.time()

        results = await self.repository.search(
            query=query,
            limit=limit,
            locale=locale,
            category=category,
            tenant_id=tenant_id,
        )

        took_ms = int((time.time() - start_time) * 1000)

        # 转换为搜索结果模型
        search_results = [
            self._to_search_result(faq, rank=idx + 1)
            for idx, faq in enumerate(results)
        ]

        return FAQSearchResponse(
            query=query,
            results=search_results,
            total=len(search_results),
            took_ms=took_ms,
        )

    # ============== 反馈功能 ==============

    async def add_feedback(
        self,
        faq_id: int,
        helpful: bool,
    ) -> FAQFeedbackResponse:
        """添加 FAQ 反馈

        Args:
            faq_id: FAQ ID
            helpful: 是否有用

        Returns:
            FAQFeedbackResponse 反馈响应

        Raises:
            HTTPException: FAQ 不存在时返回 404
        """
        faq = await self.repository.get(faq_id)

        if not faq:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ not found",
            )

        success = await self.repository.add_feedback(faq_id, helpful)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add feedback",
            )

        return FAQFeedbackResponse(
            success=True,
            helpful_count=faq.helpful_count,
            not_helpful_count=faq.not_helpful_count,
            message="Feedback recorded" if helpful else "Thank you for your feedback",
        )

    # ============== 统计功能 ==============

    async def get_stats(
        self,
        tenant_id: int | None = None,
    ) -> FAQStatsResponse:
        """获取 FAQ 统计

        Args:
            tenant_id: 租户 ID

        Returns:
            FAQStatsResponse 统计响应
        """
        # 总数
        total = await self.repository.count(
            **({"tenant_id": tenant_id} if tenant_id else {})
        )

        # 按状态统计
        by_status = await self.repository.count_by_status(tenant_id)

        # 按分类统计
        by_category = await self.repository.count_by_category(tenant_id)

        # 按语言统计（从数据库获取）
        from sqlalchemy import distinct

        statement = (
            select(FAQ.locale, func.count(FAQ.id))
            .group_by(FAQ.locale)
            .where(FAQ.status == FAQStatus.PUBLISHED)
        )

        if tenant_id is not None:
            from sqlalchemy import or_

            statement = statement.where(
                or_(FAQ.tenant_id == tenant_id, FAQ.tenant_id.is_(None))
            )

        result = await self.session.execute(statement)
        by_locale = {locale: count for locale, count in result.all()}

        # 浏览最多
        most_viewed = [
            {"id": f.id, "question": f.question, "view_count": f.view_count}
            for f in await self.repository.get_most_viewed(tenant_id=tenant_id)
        ]

        # 最有用
        most_helpful = [
            {
                "id": f.id,
                "question": f.question,
                "helpful_count": f.helpful_count,
                "not_helpful_count": f.not_helpful_count,
            }
            for f in await self.repository.get_most_helpful(tenant_id=tenant_id)
        ]

        return FAQStatsResponse(
            total=total,
            by_status=by_status,
            by_category=by_category,
            by_locale=by_locale,
            most_viewed=most_viewed,
            most_helpful=most_helpful,
        )

    # ============== 批量操作 ==============

    async def bulk_update_status(
        self,
        faq_ids: list[int],
        status: FAQStatus,
        user_id: int,
    ) -> FAQBulkUpdateResponse:
        """批量更新 FAQ 状态

        Args:
            faq_ids: FAQ ID 列表
            status: 新状态
            user_id: 用户 ID

        Returns:
            FAQBulkUpdateResponse 批量更新响应
        """
        updated_count = 0
        failed_ids: list[int] = []

        for faq_id in faq_ids:
            try:
                faq = await self.repository.get(faq_id)
                if faq:
                    data = FAQUpdate(status=status)
                    await self.repository.update_faq(faq, data, updated_by=user_id)
                    updated_count += 1
                else:
                    failed_ids.append(faq_id)
            except Exception:
                failed_ids.append(faq_id)

        logger.info(
            "faq_bulk_status_updated",
            updated_count=updated_count,
            failed_count=len(failed_ids),
        )

        return FAQBulkUpdateResponse(
            success=updated_count > 0,
            updated_count=updated_count,
            failed_ids=failed_ids,
            message=f"Updated {updated_count} FAQs"
            if updated_count > 0
            else "No FAQs were updated",
        )

    async def reorder_faqs(
        self,
        id_orders: dict[int, int],
        user_id: int,
    ) -> FAQBulkUpdateResponse:
        """重新排序 FAQ

        Args:
            id_orders: FAQ ID 到优先级的映射
            user_id: 用户 ID

        Returns:
            FAQBulkUpdateResponse 批量更新响应
        """
        updated_count = 0
        failed_ids: list[int] = []

        for faq_id, priority in id_orders.items():
            try:
                faq = await self.repository.get(faq_id)
                if faq:
                    data = FAQUpdate(priority=priority)
                    await self.repository.update_faq(faq, data, updated_by=user_id)
                    updated_count += 1
                else:
                    failed_ids.append(faq_id)
            except Exception:
                failed_ids.append(faq_id)

        logger.info(
            "faq_reordered",
            updated_count=updated_count,
            failed_count=len(failed_ids),
        )

        return FAQBulkUpdateResponse(
            success=updated_count > 0,
            updated_count=updated_count,
            failed_ids=failed_ids,
            message=f"Reordered {updated_count} FAQs"
            if updated_count > 0
            else "No FAQs were reordered",
        )


# ============== 依赖注入工厂 ==============


def get_faq_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> FAQService:
    """创建 FAQ 服务实例

    Args:
        session: 数据库会话

    Returns:
        FAQService 实例
    """
    return FAQService(session)
