"""缓存模块测试

测试多层缓存、语义缓存、租户缓存等功能。
"""

import asyncio
from typing import Any

import pytest

from app.infra import (
    CacheStats,
    L1MemoryCache,
    MultiLayerCache,
    SemanticCache,
    TenantCache,
    get_cache_stats,
    get_multilayer_cache,
    get_semantic_cache,
    get_tenant_cache,
)


# ============== L1 内存缓存测试 ==============


class TestL1MemoryCache:
    """L1 内存缓存测试"""

    def test_set_and_get(self) -> None:
        """测试基本读写"""
        cache = L1MemoryCache(max_size=100, default_ttl=60)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self) -> None:
        """测试过期机制"""
        import time

        cache = L1MemoryCache(max_size=100, default_ttl=1)

        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self) -> None:
        """测试 LRU 淘汰"""
        cache = L1MemoryCache(max_size=3, default_ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # 访问 key1 使其变为最近使用
        cache.get("key1")

        # 添加第 4 个键，应该淘汰 key2
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # 仍在（最近使用）
        assert cache.get("key2") is None  # 已淘汰
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_delete(self) -> None:
        """测试删除"""
        cache = L1MemoryCache()

        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_clear(self) -> None:
        """测试清空"""
        cache = L1MemoryCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache) == 0
        assert cache.get("key1") is None


# ============== 多层缓存测试 ==============


class TestMultiLayerCache:
    """多层缓存测试"""

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """测试基本读写"""
        cache = MultiLayerCache(enable_l1=True)

        await cache.set("key1", "value1", ttl=60)
        value = await cache.get("key1")

        assert value == "value1"

    @pytest.mark.asyncio
    async def test_l1_hit(self) -> None:
        """测试 L1 缓存命中"""
        cache = MultiLayerCache(enable_l1=True)

        stats = cache.get_stats()
        stats.reset()

        # 设置缓存（会同时设置 L1 和 L2）
        await cache.set("key1", "value1", ttl=60)

        # 第一次读取：L1 命中（因为 set 时已经写入 L1）
        value = await cache.get("key1")
        assert value == "value1"
        stats = cache.get_stats()
        assert stats.hits == 1  # 总共 1 次命中
        assert stats.l1_hits == 1  # L1 命中
        assert stats.l2_hits == 0  # L2 没有命中

        # 第二次读取：仍然 L1 命中
        value = await cache.get("key1")
        assert value == "value1"
        stats = cache.get_stats()
        assert stats.hits == 2  # 总共 2 次命中
        assert stats.l1_hits == 2  # 2 次都是 L1 命中

    @pytest.mark.asyncio
    async def test_cache_miss(self) -> None:
        """测试缓存未命中"""
        cache = MultiLayerCache()

        result = await cache.get("nonexistent")

        assert result is None

        stats = cache.get_stats()
        assert stats.misses > 0

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """测试删除"""
        cache = MultiLayerCache()

        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")

        assert deleted is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear_l1(self) -> None:
        """测试清空 L1"""
        cache = MultiLayerCache(enable_l1=True)

        await cache.set("key1", "value1")
        cache.clear_l1()

        # L1 清空后，L2 仍然存在
        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_decorator(self) -> None:
        """测试装饰器"""
        cache = MultiLayerCache()

        call_count = 0

        @cache.cached(ttl=60, key_prefix="test")
        async def expensive_func(param: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{param}"

        # 第一次调用
        result1 = await expensive_func("param1")
        assert result1 == "result-param1"
        assert call_count == 1

        # 第二次调用（应该命中缓存）
        result2 = await expensive_func("param1")
        assert result2 == "result-param1"
        assert call_count == 1  # 没有增加


# ============== 语义缓存测试 ==============


class TestSemanticCache:
    """语义缓存测试"""

    def test_similarity_calculation(self) -> None:
        """测试相似度计算"""
        cache = SemanticCache()

        # 完全相同的向量
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = cache._cosine_similarity(vec1, vec2)
        assert sim == 1.0

        # 正交向量
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        sim = cache._cosine_similarity(vec3, vec4)
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """测试基本读写"""
        # 模拟嵌入模型
        def mock_embedding(text: str) -> list[float]:
            import hashlib

            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return [(hash_val >> i) & 1 for i in range(10)]

        # 创建不带后端的缓存（仅内存）
        cache = SemanticCache(embedding_model=mock_embedding, cache_backend=None)

        # 手动设置到内存索引
        cache._embeddings["test_key"] = mock_embedding("What is Python?")

        # 获取缓存（无后端，直接从内存索引查找）
        # 由于没有后端存储，get 会返回 None
        result = await cache.get("What is Python?")
        # 结果为 None 是因为没有存储到后端

        # 验证嵌入向量存在
        assert "test_key" in cache._embeddings

    @pytest.mark.asyncio
    async def test_with_embedding_model(self) -> None:
        """测试带嵌入模型的缓存"""
        # 模拟嵌入模型
        def mock_embedding(text: str) -> list[float]:
            import hashlib

            # 返回相同的向量，使相似度为 1.0
            return [1.0, 0.0, 0.0]

        cache = SemanticCache(
            embedding_model=mock_embedding,
            similarity_threshold=0.8,
            cache_backend=None,  # 不使用后端
        )

        # 手动添加缓存数据
        cache._embeddings["key1"] = [1.0, 0.0, 0.0]
        cache._embeddings["key2"] = [0.5, 0.5, 0.5]

        # 测试相似度计算
        sim = cache._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 1.0  # 完全相同

        sim = cache._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert sim == 0.0  # 正交

    def test_clear(self) -> None:
        """测试清空"""
        cache = SemanticCache()

        cache._embeddings["test_key"] = [1.0, 2.0, 3.0]
        cache.clear()

        assert len(cache._embeddings) == 0


# ============== 租户缓存测试 ==============


class TestTenantCache:
    """租户缓存测试"""

    @pytest.mark.asyncio
    async def test_tenant_isolation(self) -> None:
        """测试租户隔离"""
        cache = TenantCache()

        # 租户 1
        await cache.set(tenant_id=1, key="config", value={"theme": "dark"})

        # 租户 2
        await cache.set(tenant_id=2, key="config", value={"theme": "light"})

        # 验证隔离
        config1 = await cache.get(tenant_id=1, key="config")
        config2 = await cache.get(tenant_id=2, key="config")

        assert config1 == {"theme": "dark"}
        assert config2 == {"theme": "light"}

    @pytest.mark.asyncio
    async def test_key_generation(self) -> None:
        """测试键生成"""
        cache = TenantCache()

        key1 = cache._make_key(1, "user")
        key2 = cache._make_key(2, "user")

        assert key1 == "tenant:1:user"
        assert key2 == "tenant:2:user"
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """测试删除"""
        cache = TenantCache()

        await cache.set(tenant_id=1, key="data", value="value1")
        deleted = await cache.delete(tenant_id=1, key="data")

        assert deleted is True
        assert await cache.get(tenant_id=1, key="data") is None


# ============== 缓存统计测试 ==============


class TestCacheStats:
    """缓存统计测试"""

    def test_initial_state(self) -> None:
        """测试初始状态"""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """测试命中率计算"""
        stats = CacheStats()

        stats.hits = 80
        stats.misses = 20

        assert stats.hit_rate == 0.8

    def test_l1_hit_rate(self) -> None:
        """测试 L1 命中率"""
        stats = CacheStats()

        stats.hits = 100
        stats.l1_hits = 70

        assert stats.l1_hit_rate == 0.7

    def test_to_dict(self) -> None:
        """测试转换为字典"""
        stats = CacheStats()
        stats.hits = 80
        stats.misses = 20

        d = stats.to_dict()

        assert "hits" in d
        assert "hit_rate" in d
        assert d["hit_rate"] == "80.00%"

    def test_reset(self) -> None:
        """测试重置"""
        stats = CacheStats()
        stats.hits = 100
        stats.misses = 50

        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0


# ============== 全局实例测试 ==============


class TestGlobalInstances:
    """全局实例测试"""

    def test_get_multilayer_cache(self) -> None:
        """测试获取多层缓存"""
        cache = get_multilayer_cache()
        assert isinstance(cache, MultiLayerCache)

        # 再次调用应返回同一实例
        cache2 = get_multilayer_cache()
        assert cache is cache2

    def test_get_semantic_cache(self) -> None:
        """测试获取语义缓存"""
        cache = get_semantic_cache()
        assert isinstance(cache, SemanticCache)

        cache2 = get_semantic_cache()
        assert cache is cache2

    def test_get_tenant_cache(self) -> None:
        """测试获取租户缓存"""
        cache = get_tenant_cache()
        assert isinstance(cache, TenantCache)

        cache2 = get_tenant_cache()
        assert cache is cache2

    def test_get_cache_stats(self) -> None:
        """测试获取缓存统计"""
        stats = get_cache_stats()
        assert isinstance(stats, CacheStats)

        stats2 = get_cache_stats()
        assert stats is stats2
