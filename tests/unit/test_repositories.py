"""Repository 层测试

测试 BaseRepository 的通用 CRUD 操作。
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import Field, SQLModel

from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams


class MockModel(SQLModel, table=True):
    """Mock 模型类 - 继承 SQLModel table 以支持 SQLAlchemy"""

    __tablename__ = "mock_model"

    id: int | None = Field(default=None, primary_key=True)
    name: str

    def __repr__(self) -> str:
        return f"MockModel(id={self.id}, name={self.name})"


class TestPaginationParams:
    """PaginationParams 测试"""

    def test_default_values(self) -> None:
        """测试默认值"""
        params = PaginationParams()
        assert params.page == 1
        assert params.size == 20

    def test_custom_values(self) -> None:
        """测试自定义值"""
        params = PaginationParams(page=3, size=50)
        assert params.page == 3
        assert params.size == 50

    def test_offset_calculation(self) -> None:
        """测试偏移量计算"""
        params = PaginationParams(page=2, size=20)
        assert params.offset == 20  # (2 - 1) * 20

    def test_limit_property(self) -> None:
        """测试限制属性"""
        params = PaginationParams(size=150)
        assert params.limit == 100  # 最大 100

        params = PaginationParams(size=50)
        assert params.limit == 50


class TestPaginatedResult:
    """PaginatedResult 测试"""

    def test_create_with_full_page(self) -> None:
        """测试创建完整页的结果"""
        params = PaginationParams(page=1, size=10)
        items = [MockModel(id=i, name=f"item{i}") for i in range(10)]

        result = PaginatedResult.create(items, 25, params)

        assert len(result.items) == 10
        assert result.total == 25
        assert result.page == 1
        assert result.size == 10
        assert result.pages == 3  # (25 + 10 - 1) // 10

    def test_create_with_partial_page(self) -> None:
        """测试创建部分页的结果"""
        params = PaginationParams(page=1, size=10)
        items = [MockModel(id=i, name=f"item{i}") for i in range(5)]

        result = PaginatedResult.create(items, 5, params)

        assert len(result.items) == 5
        assert result.pages == 1

    def test_create_empty(self) -> None:
        """测试创建空结果"""
        params = PaginationParams(page=1, size=10)

        result = PaginatedResult.create([], 0, params)

        assert len(result.items) == 0
        assert result.total == 0
        assert result.pages == 0


class TestBaseRepository:
    """BaseRepository 测试"""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Mock 数据库会话"""
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.refresh = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session) -> BaseRepository:
        """创建测试仓储实例"""
        return BaseRepository(MockModel, mock_session)

    def test_initialization(self, repository, mock_session) -> None:
        """测试初始化"""
        assert repository.model is MockModel
        assert repository.session is mock_session

    @pytest.mark.asyncio
    async def test_get_found(self, repository, mock_session) -> None:
        """测试获取存在的记录"""
        mock_model = MockModel(id=1, name="test")

        # 模拟 execute 返回结果
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
        mock_session.execute.return_value = mock_result

        result = await repository.get(1)
        assert result is mock_model
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_get_not_found(self, repository, mock_session) -> None:
        """测试获取不存在的记录"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = mock_result

        result = await repository.get(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by(self, repository, mock_session) -> None:
        """测试根据条件获取"""
        mock_model = MockModel(id=1, name="test")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_model)
        mock_session.execute.return_value = mock_result

        result = await repository.get_by(name="test")
        assert result is mock_model

    @pytest.mark.asyncio
    async def test_list(self, repository, mock_session) -> None:
        """测试列表查询"""
        mock_models = [MockModel(id=i, name=f"item{i}") for i in range(3)]

        # 模拟 result.scalars().all() 链式调用
        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=mock_models)

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        mock_session.execute.return_value = mock_result

        result = await repository.list(offset=0, limit=10)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_with_filter(self, repository, mock_session) -> None:
        """测试带过滤条件的列表查询"""
        mock_models = [MockModel(id=1, name="test")]

        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=mock_models)

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        mock_session.execute.return_value = mock_result

        result = await repository.list(name="test")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_paginated(self, repository, mock_session) -> None:
        """测试分页查询"""
        mock_models = [MockModel(id=i, name=f"item{i}") for i in range(10)]

        # 第一次调用获取总数
        mock_count_result = MagicMock()
        mock_count_result.scalar = MagicMock(return_value=25)

        # 第二次调用获取数据
        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=mock_models)
        mock_items_result = MagicMock()
        mock_items_result.scalars = MagicMock(return_value=mock_scalars)

        mock_session.execute.side_effect = [mock_count_result, mock_items_result]

        params = PaginationParams(page=1, size=10)
        result = await repository.list_paginated(params)

        assert isinstance(result, PaginatedResult)
        assert result.total == 25
        assert len(result.items) == 10

    @pytest.mark.asyncio
    async def test_create(self, repository, mock_session) -> None:
        """测试创建记录"""
        new_model = MockModel(id=1, name="new")  # 模拟数据库已分配 ID

        result = await repository.create(new_model)

        mock_session.add.assert_called_once_with(new_model)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(new_model)

    @pytest.mark.asyncio
    async def test_create_failure_rollback(self, repository, mock_session) -> None:
        """测试创建失败时回滚"""
        new_model = MockModel(id=None, name="new")
        mock_session.commit.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await repository.create(new_model)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update(self, repository, mock_session) -> None:
        """测试更新记录"""
        existing_model = MockModel(id=1, name="old")

        # 模拟 get() 调用
        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none = MagicMock(return_value=existing_model)

        # 模拟 update 中的 commit 调用
        mock_session.execute.return_value = mock_get_result

        # 这里需要模拟 get() 方法的行为
        # 由于 update() 内部调用 get()，我们需要让 get() 返回 existing_model
        original_get = repository.get
        async def mock_get(id_val):
            if id_val == 1:
                return existing_model
            return await original_get(id_val)

        repository.get = mock_get

        result = await repository.update(1, name="new")

        assert result.name == "new"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(self, repository, mock_session) -> None:
        """测试更新不存在的记录"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = mock_result

        result = await repository.update(999, name="new")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, repository, mock_session) -> None:
        """测试删除记录"""
        existing_model = MockModel(id=1, name="test")

        # 模拟 get() 调用
        original_get = repository.get
        async def mock_get(id_val):
            if id_val == 1:
                return existing_model
            return await original_get(id_val)

        repository.get = mock_get

        result = await repository.delete(1)

        assert result is True
        mock_session.delete.assert_called_once_with(existing_model)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, repository, mock_session) -> None:
        """测试删除不存在的记录"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = mock_result

        result = await repository.delete(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_count(self, repository, mock_session) -> None:
        """测试统计记录"""
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=42)
        mock_session.execute.return_value = mock_result

        result = await repository.count()
        assert result == 42

    @pytest.mark.asyncio
    async def test_count_with_filter(self, repository, mock_session) -> None:
        """测试带过滤条件的统计"""
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=5)
        mock_session.execute.return_value = mock_result

        result = await repository.count(name="test")
        assert result == 5

    @pytest.mark.asyncio
    async def test_exists_true(self, repository, mock_session) -> None:
        """测试记录存在"""
        # exists() 内部调用 count()
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=1)
        mock_session.execute.return_value = mock_result

        result = await repository.exists(name="test")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, repository, mock_session) -> None:
        """测试记录不存在"""
        mock_result = MagicMock()
        mock_result.scalar = MagicMock(return_value=0)
        mock_session.execute.return_value = mock_result

        result = await repository.exists(name="test")
        assert result is False


@pytest.mark.parametrize("page,size,expected_offset", [
    (1, 10, 0),
    (2, 10, 10),
    (3, 20, 40),
    (5, 50, 200),
])
def test_pagination_offset_calculation(page: int, size: int, expected_offset: int) -> None:
    """参数化测试分页偏移量计算"""
    params = PaginationParams(page=page, size=size)
    assert params.offset == expected_offset


@pytest.mark.parametrize("total,size,expected_pages", [
    (0, 10, 0),
    (5, 10, 1),
    (10, 10, 1),
    (11, 10, 2),
    (25, 10, 3),
    (100, 20, 5),
])
def test_paginated_result_pages_calculation(total: int, size: int, expected_pages: int) -> None:
    """参数化测试分页页数计算"""
    params = PaginationParams(page=1, size=size)
    result = PaginatedResult.create([], total, params)
    assert result.pages == expected_pages
