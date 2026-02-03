# 缓存模块使用指南

Kiki 企业级缓存系统提供多层缓存、语义缓存、租户隔离等功能。

## 功能概览

| 功能 | 说明 | 适用场景 |
|------|------|----------|
| **RedisCache** | Redis 异步缓存 | 通用缓存 |
| **MultiLayerCache** | L1(内存) + L2(Redis) 双层缓存 | 高频热点数据 |
| **SemanticCache** | 基于向量相似度的智能缓存 | LLM 响应缓存 |
| **TenantCache** | 租户隔离缓存 | 多租户 SaaS |
| **DistributedLock** | Redis 分布式锁 | 防止缓存击穿 |
| **CacheStats** | 缓存统计监控 | 性能分析 |

---

## 基础用法

### 1. Redis 缓存

```python
from app.infra import cache_instance

# 设置缓存
await cache_instance.set("user:123", {"name": "Alice", "age": 25}, ttl=600)

# 获取缓存
user = await cache_instance.get("user:123")

# 删除缓存
await cache_instance.delete("user:123")

# 批量操作
await cache_instance.set_many({
    "user:1": {"name": "Alice"},
    "user:2": {"name": "Bob"},
}, ttl=600)
users = await cache_instance.get_many(["user:1", "user:2"])
```

### 2. 装饰器缓存

```python
from app.infra import cached

@cached(ttl=600, key_prefix="user")
async def get_user(user_id: int):
    return await db.fetch_user(user_id)

# 强制跳过缓存
user = await get_user(123, _cache_bypass=True)
```

---

## 多层缓存

L1 内存缓存用于热点数据，减少 Redis 网络往返。

```python
from app.infra import get_multilayer_cache

cache = get_multilayer_cache()

# 读取先查 L1，再查 L2，自动回填 L1
user = await cache.get("user:123")

# 写入同时写入 L1 和 L2
await cache.set("user:123", {"name": "Alice"}, ttl=600)

# 使用装饰器
@cache.cached(ttl=600, key_prefix="user", use_l1=True)
async def get_user(user_id: int):
    return await db.fetch_user(user_id)
```

### 查看缓存统计

```python
from app.infra import get_cache_stats

stats = get_cache_stats()
print(stats.to_dict())
# {
#     "hits": 1500,
#     "misses": 300,
#     "hit_rate": "83.33%",
#     "l1_hits": 1000,
#     "l1_hit_rate": "66.67%",
#     ...
# }
```

---

## 语义缓存

基于嵌入向量相似度的智能缓存，用于 LLM 响应。

```python
from app.infra import get_semantic_cache
from app.llm import get_embeddings_model

# 获取嵌入模型
embeddings = get_embeddings_model()
cache = get_semantic_cache()
cache.embedding_model = embeddings.embed_query

# 设置缓存
await cache.set(
    query="What is Python?",
    response="Python is a high-level programming language...",
    ttl=3600,
)

# 语义相似的查询会命中缓存
response = await cache.get("Tell me about Python")  # 会命中！
```

---

## 租户隔离缓存

为每个租户提供独立的缓存命名空间。

```python
from app.infra import get_tenant_cache

cache = get_tenant_cache()

# 租户 1
await cache.set(tenant_id=1, key="config", value={"theme": "dark"})
config = await cache.get(tenant_id=1, key="config")

# 租户 2（相同键不会冲突）
await cache.set(tenant_id=2, key="config", value={"theme": "light"})
config = await cache.get(tenant_id=2, key="config")  # {"theme": "light"}

# 清空租户的所有缓存
await cache.clear_tenant(tenant_id=1)
```

---

## 防止缓存问题

### 缓存雪崩（TTL 抖动）

RedisCache 自动添加 ±10% 的 TTL 抖动，防止同时过期。

```python
from app.infra import RedisCache

cache = RedisCache(jitter_percent=0.1)  # 默认已启用
```

### 缓存击穿（分布式锁）

```python
from app.infra import DistributedLock

lock = DistributedLock(cache_instance)

async def get_hot_data(key: str):
    # 获取锁
    if await lock.acquire(key, timeout=10):
        try:
            # 二次检查
            cached = await cache_instance.get(key)
            if cached:
                return cached

            # 查询数据库
            data = await db.query(key)
            await cache_instance.set(key, data, ttl=600)
            return data
        finally:
            await lock.release(key)
```

### 缓存穿透（空值缓存）

```python
from app.infra import CachePenetrationProtection

protection = CachePenetrationProtection(cache_instance, null_ttl=60)

async def get_user(user_id: int):
    return await protection.get_or_fetch(
        key=f"user:{user_id}",
        fetch_func=lambda: db.fetch_user(user_id),
        ttl=600,
    )
```

---

## 性能优化建议

1. **热点数据使用多层缓存**
   - L1 缓存：1000 条，60 秒 TTL
   - L2 缓存：根据业务设置

2. **LLM 响应使用语义缓存**
   - 相似度阈值：0.85-0.90
   - 可减少 30-50% 的 LLM 调用

3. **监控缓存命中率**
   - 目标命中率：>80%
   - L1 命中率：>60%

4. **批量操作**
   - 使用 `set_many` / `get_many`
   - 减少网络往返

---

## 配置

在 `app/config/settings.py` 中配置：

```python
class Settings(BaseSettings):
    # Redis 连接
    redis_url: str = "redis://localhost:6379/0"

    # 缓存配置
    cache_default_ttl: int = 300
    cache_l1_max_size: int = 1000
    cache_l1_ttl: int = 60

    # 语义缓存
    semantic_similarity_threshold: float = 0.85
```
