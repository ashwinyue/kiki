# Kiki 可观测性与高并发增强文档

本文档介绍基于 AI 工程师训练营参考实现的生产级增强功能。

> **路线图：** 查看后续实现计划，请参阅 [ROADMAP.md](./ROADMAP.md)

---

## 📁 新增文件清单

### 核心模块

| 文件 | 描述 | 参考 |
|------|------|------|
| `app/observability/elk_handler.py` | ELK Logstash TCP 日志处理器 | week08/p41elk.py |
| `app/core/token_bucket.py` | 令牌桶限流中间件 | week09/3/p29限流中间件 |
| `app/infra/cache.py` | Redis 缓存基础设施（TTL 抖动、分布式锁、穿透防护） | week09/3/p30缓存策略 |
| `app/agent/memory/window.py` | 窗口记忆（Token 限制） | week07/p07-windowMEM.py |
| `app/agent/retry.py` | 工具重试机制 | week07/p13-toolRetry.py |

### 配置文件

| 文件 | 描述 |
|------|------|
| `config/prometheus.yml` | Prometheus 抓取配置 |
| `config/alerts/kiki_alerts.yml` | Prometheus 告警规则 |
| `config/logstash/logstash.conf` | Logstash 管道配置 |
| `config/alertmanager.yml` | Alertmanager 配置 |
| `grafana/dashboards/kiki-dashboard.json` | Grafana 仪表板 |
| `docker-compose.observability.yml` | 可观测性服务栈 |

---

## 🚀 快速启动

### 1. 启动可观测性服务栈

```bash
# 启动 ELK + Prometheus + Grafana
docker-compose -f docker-compose.observability.yml up -d

# 访问地址
# - Kibana: http://localhost:5601
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### 2. 启用 ELK 日志

在 `.env` 文件中配置：

```bash
KIKI_ELK_ENABLED=true
KIKI_ELK_HOST=localhost
KIKI_ELK_PORT=5044
```

然后在 `app/main.py` 中注册：

```python
from app.observability.elk_handler import setup_elk_logging

# 在应用启动时调用
setup_elk_logging(logger)
```

### 3. 启用令牌桶限流

在 `app/main.py` 中添加中间件：

```python
from app.core.token_bucket import TokenBucketRateLimiter

app.add_middleware(
    TokenBucketRateLimiter,
    rate_per_sec=10.0,      # 10 令牌/秒
    burst_capacity=50,      # 突发容量 50
    exempt_paths={"/health", "/metrics", "/docs"},
)
```

### 5. 使用增强缓存

```python
from app.infra.cache import cached, cache_instance

# 装饰器方式
@cached(ttl=600, key_prefix="user")
async def get_user(user_id: int) -> User:
    return await db.fetch_user(user_id)

# 强制跳过缓存
user = await get_user(123, _cache_bypass=True)

# 直接使用缓存实例
await cache_instance.set("key", value, ttl=300)
value = await cache_instance.get("key")
```

---

## 📊 功能详解

### 1. ELK 日志处理器

**特性：**
- 自动重连机制（最多 3 次重试）
- 线程安全（使用 RLock）
- 批量发送模式
- 降级到本地文件

**使用方式：**

```python
from app.observability.elk_handler import ELKHandler, BatchELKHandler

# 基础处理器
handler = ELKHandler(
    host="localhost",
    port=5044,
    timeout=5.0,
    max_retries=3,
    enable_fallback=True,
)
logger.addHandler(handler)

# 批量处理器（推荐生产环境）
batch_handler = BatchELKHandler(
    host="localhost",
    port=5044,
    batch_size=10,          # 累积 10 条后发送
    batch_timeout=5.0,      # 或 5 秒后发送
)
logger.addHandler(batch_handler)
```

**日志格式：**

```json
{
  "@timestamp": "2025-01-31T10:30:45.123Z",
  "level": "INFO",
  "logger_name": "app.api.chat",
  "message": "处理聊天请求",
  "thread": 12345,
  "process": 6789,
  "source": {
    "file": "/app/api/chat.py",
    "line": 42,
    "function": "handle_chat"
  },
  "context": {
    "request_id": "abc-123",
    "user_id": "user-001",
    "session_id": "sess-xyz"
  }
}
```

---

### 2. 令牌桶限流

**与 slowapi 对比：**

| 特性 | slowapi (Kiki 原有) | TokenBucket (新增) |
|------|---------------------|-------------------|
| 算法 | 固定窗口 | 令牌桶 |
| 突发支持 | ❌ | ✅ |
| 自定义键 | ✅ | ✅ |
| 响应头 | 基础 | 完整 (RFC 6585) |
| 分布式 | Redis | 内存 (可扩展) |

**使用方式：**

```python
from app.core.token_bucket import (
    TokenBucketRateLimiter,
    PathBasedRateLimiter,
    RateLimitPolicy,
)

# 方式 1: 全局中间件
app.add_middleware(
    TokenBucketRateLimiter,
    rate_per_sec=10.0,
    burst_capacity=50,
    exempt_paths={"/health", "/docs"},
)

# 方式 2: 基于路径的不同策略
app.add_middleware(
    PathBasedRateLimiter,
    policies={
        "/api/v1/chat": RateLimitPolicy(rate=0.5, burst_capacity=10),   # 2 req/s
        "/api/v1/agents": RateLimitPolicy(rate=2.0, burst_capacity=20),  # 10 req/s
    },
    default_policy=RateLimitPolicy(rate=10.0, burst_capacity=50),
)

# 方式 3: 自定义键（按用户限流）
def user_key_func(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    return f"user:{user_id}" if user_id else f"ip:{request.client.host}"

app.add_middleware(
    TokenBucketRateLimiter,
    rate_per_sec=5.0,
    burst_capacity=20,
    key_func=user_key_func,
)
```

**响应头：**

```
X-RateLimit-Policy: token_bucket; rate=10.0/s; burst=50
X-RateLimit-Limit: 50
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 3
Retry-After: 3  # 仅限流时出现
```

---

---

### 4. 增强缓存

**TTL 抖动：**

```python
from app.infra.cache import RedisCache

cache = RedisCache(
    redis_url="redis://localhost:6379/0",
    default_ttl=300,
    jitter_percent=0.1,  # ±10% 抖动
)

# 设置缓存（自动添加抖动）
await cache.set("key", "value", ttl=300)
# 实际 TTL 可能是 270 ~ 330 秒
```

**分布式锁：**

```python
from app.infra.cache import DistributedLock

lock = DistributedLock(cache)

# 方式 1: 手动管理
acquired = await lock.acquire("resource_name", timeout=10)
if acquired:
    try:
        # 执行需要保护的操作
        result = await expensive_operation()
    finally:
        await lock.release("resource_name")

# 方式 2: 上下文管理器
async with lock:
    result = await expensive_operation()
```

**缓存穿透防护：**

```python
from app.infra.cache import CachePenetrationProtection

protection = CachePenetrationProtection(cache, null_ttl=60)

async def fetch_user(user_id: int):
    # 数据库查询函数
    return await db.query(user_id)

# 自动处理空值缓存
result = await protection.get_or_fetch(
    f"user:{user_id}",
    fetch_user,
    ttl=300,
)
```

**缓存装饰器：**

```python
from app.infra.cache import cached

@cached(
    ttl=600,                    # 缓存 10 分钟
    key_prefix="prediction",     # 键前缀
    exclude_params=["debug"],    # 排除 debug 参数
)
async def get_prediction(model: str, features: list, debug: bool = False):
    # 计算...
    return prediction

# 强制跳过缓存
result = await get_prediction("model", [1,2,3], _cache_bypass=True)
```

---

## 📈 Grafana 仪表板

导入仪表板：`grafana/dashboards/kiki-dashboard.json`

**包含的面板：**

1. **请求速率 (QPS)** - 按方法和状态码分组
2. **P95 响应时间** - 实时 gauge
3. **API 错误率** - 百分比趋势
4. **活跃连接数** - 会话和 WebSocket
5. **LLM 请求速率** - 按模型和状态
6. **Agent 执行耗时** - P50/P95 分位数
7. **内存使用率** - 百分比 gauge
8. **CPU 使用率** - 百分比 gauge
9. **Redis 连接数** - 实时连接数
10. **内存使用量** - 字节数

---

## 🚨 告警规则

**预定义告警：**

| 告警名称 | 条件 | 级别 |
|----------|------|------|
| HighErrorRate | 错误率 > 5% (5分钟) | warning |
| SlowResponseTime | P95 > 1秒 (10分钟) | warning |
| HighLLMErrorRate | LLM 失败率 > 10% (5分钟) | critical |
| HighMemoryUsage | 内存使用率 > 80% (10分钟) | warning |
| HighAgentFailureRate | Agent 失败率 > 20% (10分钟) | warning |

---

## 🔧 配置项

### 环境变量

```bash
# ELK 配置
KIKI_ELK_ENABLED=true
KIKI_ELK_HOST=localhost
KIKI_ELK_PORT=5044
KIKI_ELK_TIMEOUT=5.0
KIKI_ELK_MAX_RETRIES=3
KIKI_ELB_FALLBACK_ENABLED=true

# 限流配置
KIKI_RATE_LIMIT_ENABLED=true
KIKI_RATE_LIMIT_DEFAULT_RATE=10.0
KIKI_RATE_LIMIT_DEFAULT_BURST=50
```

---

## 📝 最佳实践

### 1. 日志记录

```python
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 结构化日志
logger.info(
    "user_login",
    user_id=user.id,
    ip=request.client.host,
)

# 异常日志
try:
    ...
except Exception as e:
    logger.exception(
        "operation_failed",
        operation="create_agent",
        user_id=user.id,
    )
```

### 2. 限流策略

```python
# 聊天接口：低频次，允许中等突发
"/chat": rate=0.5/s, burst=10

# API 接口：高频次，允许大突发
"/api": rate=10/s, burst=100

# 注册/登录：低频次，小突发
"/auth": rate=0.1/s, burst=5
```

### 3. 缓存策略

```python
# 热数据：长 TTL，小抖动
user_profile: ttl=3600, jitter=5%

# 温数据：中等 TTL
search_result: ttl=300, jitter=10%

# 冷数据：短 TTL，大抖动
statistic_data: ttl=60, jitter=20%
```

### 4. WebSocket 连接管理

```python
# 心跳保活
setInterval(() => {
  ws.send(JSON.stringify({ action: 'ping' }));
}, 30000);

# 重连机制
ws.onclose = () => {
  setTimeout(() => reconnect(), 1000);
};
```

---

## 🪟 窗口记忆（Window Memory）

### 概述

窗口记忆机制基于 LangChain 的 `trim_messages` 实现 Token 级别的滑动窗口，确保对话历史不超过 LLM 的上下文限制。

**核心特性：**
- Token 级别限制（而非简单的消息数量）
- 支持多种修剪策略（last/first）
- 确保对话边界完整性
- 支持自定义 token 计数器

**参考实现：** `week07/p07-windowMEM.py`

### 基础使用

```python
from app.agent.memory import create_pre_model_hook

# 创建 pre_model_hook
hook = create_pre_model_hook(
    max_tokens=384,        # 最大 Token 数
    strategy="last",       # 保留最新的消息
    start_on="human",      # 从人类消息开始
    end_on=("human", "tool"),  # 在人类或工具消息结束
)

# 用于 LangGraph
builder = StateGraph(AgentState)
builder.add_node(
    "agent",
    model_node,
    pre_model_hook=hook,  # 添加钩子
)
```

### 高级使用

```python
from app.agent.memory import WindowMemoryManager, TokenCounterType

# 创建管理器
manager = WindowMemoryManager(
    max_tokens=1000,
    strategy="last",
    token_counter_type=TokenCounterType.APPROXIMATE,
    preserve_system=True,
)

# 修剪消息
trimmed = manager.trim_messages(messages)

# 获取统计
stats = manager.get_stats()
print(f"总修剪次数: {stats['total_trims']}")
print(f"平均移除 Token: {stats['avg_tokens_removed']}")
```

### 便捷函数

```python
from app.agent.memory import (
    create_chat_hook,      # 聊天专用钩子
    trim_state_messages,   # 直接修剪状态消息
    get_window_memory_manager,  # 获取全局管理器
)

# 聊天场景
hook = create_chat_hook(max_tokens=500)

# 直接修剪
trimmed = trim_state_messages(state, max_tokens=384)
```

---

## 🔄 工具重试（Tool Retry）

### 概述

工具重试机制基于 LangGraph 的 `retry` 参数实现自动重试，支持指数退避算法避免雪崩。

**核心特性：**
- 可配置的重试策略（次数、间隔、退避因子）
- 支持指定可重试的异常类型
- 指数退避算法
- 支持自定义重试条件

**参考实现：** `week07/p13-toolRetry.py`

### 异常类型

```python
from app.agent.retry import (
    RetryableError,          # 可重试错误基类
    NetworkError,            # 网络错误
    RateLimitError,          # 速率限制
    ResourceUnavailableError,  # 资源不可用
    TemporaryServiceError,   # 临时服务错误
    ToolExecutionError,      # 工具执行错误（不重试）
)
```

### 重试策略

```python
from app.agent.retry import RetryPolicy, RetryStrategy

policy = RetryPolicy(
    max_attempts=3,                    # 最大尝试次数
    retry_on=(NetworkError, RateLimitError),  # 可重试异常
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,  # 指数退避
    initial_interval=1.0,              # 初始间隔 1 秒
    backoff_factor=2.0,                # 退避因子
    max_interval=60.0,                 # 最大间隔 60 秒
    jitter=True,                       # 启用抖动
    jitter_percent=0.1,                # 抖动 ±10%
)
```

### 装饰器方式

```python
from app.agent.retry import with_retry

@with_retry(max_attempts=3)
async def risky_operation():
    # 可能失败的操作
    response = await api_call()
    return response

# 带重试回调
async def on_retry_fn(error, attempt):
    logger.warning(f"重试 {attempt}: {error}")

@with_retry(policy=policy, on_retry=on_retry_fn)
async def operation_with_callback():
    pass
```

### 上下文管理器

```python
from app.agent.retry import RetryContext

policy = RetryPolicy(max_attempts=3)

async with RetryContext(policy) as retry:
    result = await retry.attempt(risky_function)
```

### LangGraph 节点

```python
from app.agent.retry import create_retryable_node

async def my_tool_node(state: AgentState) -> dict:
    # 工具逻辑
    return {"messages": [...]}

# 创建可重试节点
retryable_node = create_retryable_node(
    my_tool_node,
    policy=RetryPolicy(max_attempts=3)
)

builder.add_node("my_tool", retryable_node)
```

### 便捷函数

```python
from app.agent.retry import execute_with_retry

result = await execute_with_retry(
    llm.ainvoke,
    messages,
    policy=RetryPolicy(max_attempts=3)
)
```

### 重试策略对比

| 策略 | 描述 | 延迟计算 | 适用场景 |
|------|------|----------|----------|
| IMMEDIATE | 立即重试 | 0 | 测试环境 |
| FIXED_INTERVAL | 固定间隔 | initial_interval | 稳定服务 |
| LINEAR_BACKOFF | 线性退避 | initial_interval × attempt | 轻微过载 |
| EXPONENTIAL_BACKOFF | 指数退避 | initial_interval × backoff_factor^(attempt-1) | 高并发 |

---

## 📚 参考资料

- [ELK Stack 官方文档](https://www.elastic.co/guide/)
- [Prometheus 最佳实践](https://prometheus.io/docs/practices/)
- [令牌桶算法详解](https://en.wikipedia.org/wiki/Token_bucket)
- [WebSocket RFC 6455](https://tools.ietf.org/html/rfc6455)
- [LangChain trim_messages](https://python.langchain.com/docs/how_to/trim_messages/)
- [LangGraph 重试机制](https://langchain-ai.github.io/langgraph/reference/checkpoints/#retry-policy)
