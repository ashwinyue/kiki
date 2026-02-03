# Checkpoint 流式消息持久化实现报告

> 实施日期: 2025-02-03
> 参考: DeerFlow `src/graph/checkpoint.py`

## 概述

基于 DeerFlow 的 `ChatStreamManager` 设计，为 Kiki 项目实现了流式对话消息的持久化功能。该功能使用双层存储架构：

1. **InMemoryStore** - 临时缓存流式消息块
2. **PostgreSQL** - 持久化完整对话（在 finish_reason 触发时）

## 实现的文件

### 1. 核心模块

| 文件 | 说明 |
|------|------|
| `app/agent/graph/chat_stream.py` | ChatStreamManager 类，核心实现 |
| `migrations/002_create_chat_streams.sql` | 数据库表迁移 |
| `app/agent/streaming/service.py` | 集成到流继续服务 |
| `app/config/settings.py` | 添加配置选项 |

### 2. API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/messages/streams/{thread_id}` | GET | 获取对话历史 |
| `/api/v1/messages/streams` | GET | 列出最近线程 |
| `/api/v1/messages/streams/{thread_id}` | DELETE | 删除对话历史 |

### 3. 测试文件

| 文件 | 说明 |
|------|------|
| `tests/unit/test_chat_stream.py` | 单元测试 |

## 核心功能

### ChatStreamManager 类

```python
class ChatStreamManager:
    """流式对话消息管理器

    双层存储架构：
    1. InMemoryStore - 临时缓存流式消息块
    2. PostgreSQL - 持久化完整对话
    """
```

#### 主要方法

| 方法 | 说明 |
|------|------|
| `initialize()` | 初始化管理器，创建数据库表 |
| `process_stream_message()` | 处理流式消息 |
| `_persist_to_postgresql()` | 持久化到数据库 |
| `get_chat_history()` | 获取对话历史 |
| `delete_chat_history()` | 删除对话历史 |
| `list_recent_threads()` | 列出最近线程 |
| `close()` | 关闭管理器 |

### 数据库表结构

```sql
CREATE TABLE chat_streams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    messages JSONB NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 使用示例

### 1. 基本使用

```python
from app.agent.graph.chat_stream import get_chat_stream_manager

# 获取管理器实例
manager = await get_chat_stream_manager()

# 处理流式消息
await manager.process_stream_message(
    thread_id="thread-123",
    message="Hello, world!",
    finish_reason="stop"  # 触发持久化
)

# 获取对话历史
history = await manager.get_chat_history("thread-123")
print(history)  # ["Hello, world!"]
```

### 2. 集成到流式服务

```python
from app.agent.streaming.service import get_stream_continuation_service

service = await get_stream_continuation_service()

# 注册流
await service.register_stream("session-123")

# 添加事件
await service.add_event("session-123", StreamEvent(
    event_type="token",
    content="Hello",
))

# 完成流（自动触发持久化）
await service.complete_stream("session-123")
```

### 3. API 查询

```bash
# 获取对话历史
curl http://localhost:8000/api/v1/messages/streams/thread-123

# 列出最近线程
curl http://localhost:8000/api/v1/messages/streams?limit=10

# 删除对话历史
curl -X DELETE http://localhost:8000/api/v1/messages/streams/thread-123
```

## 配置选项

在 `app/config/settings.py` 中添加的配置：

```python
# ========== 流式消息持久化配置 ==========
# 是否启用流式消息持久化
chat_stream_checkpoint_saver: bool = True
# 流式消息过期天数（用于清理）
chat_stream_retention_days: int = 30
```

## 设计决策

### 1. 双层存储架构

**参考**: DeerFlow 使用 InMemoryStore + MongoDB/PostgreSQL

**实现**:
- InMemoryStore: 临时缓存流式消息块，按索引存储
- PostgreSQL: 完整对话持久化，JSONB 格式存储消息列表

**优势**:
- 性能优化：先内存后数据库
- 灵活性：可选择何时持久化
- 可靠性：数据库保证数据不丢失

### 2. finish_reason 触发持久化

**参考**: DeerFlow 在 `finish_reason in ("stop", "interrupt")` 时持久化

**实现**:
```python
if finish_reason in ("stop", "interrupt"):
    return await self._persist_complete_conversation(...)
```

**原因**:
- `stop`: 正常完成，需要持久化完整对话
- `interrupt`: 中断完成，也需要保存已生成的内容
- 其他: 流式输出中，不需要每次都持久化

### 3. 消息块合并

**参考**: DeerFlow 使用 chunk 索引存储消息块

**实现**:
```python
# 存储消息块
chunk_key = f"chunk_{current_index}"
self.store.put(store_namespace, chunk_key, message)

# 持久化时合并
for item in self.store.search(store_namespace):
    if not isinstance(item.value, dict):  # 排除游标元数据
        messages.append(str(item.value))
```

## 测试覆盖

### 单元测试

- ✅ 消息缓存功能
- ✅ 消息持久化触发
- ✅ 多条消息处理
- ✅ 无效输入处理
- ✅ 对话历史查询
- ✅ 删除对话历史
- ✅ 最近线程列表
- ✅ 全局单例模式

### 集成测试

- ⏳ 完整工作流（需要真实数据库）
- ⏳ 并发流处理

## 后续改进

### 短期改进

1. **测试覆盖** - 添加更多集成测试
2. **性能优化** - 批量写入优化
3. **错误处理** - 更完善的错误恢复机制

### 长期改进

1. **多租户支持** - 添加 tenant_id 隔离
2. **压缩存储** - 对长消息进行压缩
3. **TTL 管理** - 自动清理过期对话
4. **分布式支持** - Redis 作为中间层

## 参考

- DeerFlow: https://github.com/bytedance/deer-flow
- LangGraph Checkpoint: https://langchain-ai.github.io/langgraph/concepts/persistence/
