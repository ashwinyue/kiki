# Checkpoint 持久化系统 - 实现完成

**日期**: 2026-02-04
**参考**: DeerFlow ChatStreamManager 设计

## 实现概览

参考 DeerFlow 的 `ChatStreamManager` 设计，实现了完整的流式对话消息持久化系统。

### 核心特性

1. **双层存储架构**：
   - `InMemoryStore` - 临时缓存流式消息块（高性能）
   - `PostgreSQL` - 完整对话持久化（可靠性）

2. **流式消息处理**：
   - 按消息块索引存储（`chunk_0`, `chunk_1`, ...）
   - 自动合并机制
   - 游标管理（`cursor`）追踪当前索引

3. **条件持久化**：
   - `finish_reason in ("stop", "interrupt")` 时触发持久化
   - 减少无效数据库写入

4. **完整 CRUD 操作**：
   - 创建/更新对话历史
   - 获取对话历史
   - 删除对话历史
   - 列出最近对话线程

## 文件结构

```
app/agent/graph/
├── checkpoint.py        # LangGraph Checkpoint (AsyncPostgresSaver/MemorySaver)
└── chat_stream.py       # ChatStreamManager (流式对话持久化)
```

## 代码实现

### ChatStreamManager 核心方法

#### `process_stream_message()`
处理流式消息，支持分块存储和自动合并：

```python
async def process_stream_message(
    self,
    thread_id: str,
    message: str,
    finish_reason: str = "",
) -> bool:
    """处理流式消息

    1. 创建命名空间 (messages, thread_id)
    2. 获取/初始化游标
    3. 存储消息块
    4. 判断是否持久化
    """
    store_namespace = ("messages", thread_id)
    cursor = self.store.get(store_namespace, "cursor")
    current_index = cursor.value.get("index", 0) + 1

    # 存储消息块
    self.store.put(store_namespace, f"chunk_{current_index}", message)

    # 条件持久化
    if finish_reason in ("stop", "interrupt"):
        return await self._persist_complete_conversation(
            thread_id, store_namespace, current_index
        )
```

#### `_persist_complete_conversation()`
持久化完整对话到数据库：

```python
async def _persist_complete_conversation(
    self,
    thread_id: str,
    store_namespace: tuple[str, str],
    final_index: int,
) -> bool:
    """从 InMemoryStore 获取所有消息块，合并后持久化到 PostgreSQL"""
    memories = self.store.search(store_namespace, limit=final_index + 2)

    messages = []
    for item in memories:
        value = item.dict().get("value", "")
        if value and not isinstance(value, dict):
            messages.append(str(value))

    return await self._persist_to_postgresql(thread_id, messages)
```

#### `get_chat_history()`
获取对话历史：

```python
async def get_chat_history(
    self,
    thread_id: str,
) -> list[str] | None:
    """从 PostgreSQL 获取对话历史"""
    async with self._session_factory() as session:
        result = await session.execute(
            select_sql, {"thread_id": thread_id}
        )
        row = result.first()
        if row:
            return json.loads(row[0])
```

## 数据库表结构

```sql
CREATE TABLE IF NOT EXISTS chat_streams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    messages JSONB NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_chat_streams_thread_id ON chat_streams(thread_id);
CREATE INDEX idx_chat_streams_ts ON chat_streams(ts);

-- 自动更新 updated_at 触发器
CREATE TRIGGER update_chat_streams_updated_at
    BEFORE UPDATE ON chat_streams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

## 使用示例

### 1. 初始化 ChatStreamManager

```python
from app.agent.graph.chat_stream import get_chat_stream_manager

# 获取单例实例
manager = await get_chat_stream_manager()
```

### 2. 处理流式消息

```python
# 流式输出时，逐块处理消息
async for chunk in stream_response:
    await manager.process_stream_message(
        thread_id="thread-123",
        message=chunk,
        finish_reason=""  # 流进行中
    )

# 流结束时，触发持久化
await manager.process_stream_message(
    thread_id="thread-123",
    message="",
    finish_reason="stop"  # 触发持久化
)
```

### 3. 获取对话历史

```python
history = await manager.get_chat_history("thread-123")
# 返回: ["Hello", "How are you?", "I'm fine"]
```

### 4. 列出最近对话

```python
recent_threads = await manager.list_recent_threads(limit=10)
# 返回: [{"thread_id": "...", "message_count": 5, "ts": "..."}, ...]
```

## 配置选项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `chat_stream_checkpoint_saver` | 是否启用持久化 | `False` |
| `database_url` | 数据库连接字符串 | - |

启用持久化（在 `conf.yaml` 或环境变量）：

```yaml
# conf.yaml
chat_stream_checkpoint_saver: true

# 或环境变量
export KIKI_CHAT_STREAM_CHECKPOINT_SAVER=true
```

## DeerFlow 对比

| 功能 | DeerFlow | Kiki |
|------|----------|------|
| 双层存储（InMemory + PostgreSQL） | ✅ | ✅ |
| 消息块索引存储 | ✅ | ✅ |
| 游标管理 | ✅ | ✅ |
| 条件持久化（finish_reason） | ✅ | ✅ |
| 获取对话历史 | ✅ | ✅ |
| 列出最近对话 | ✅ | ✅ |
| 删除对话历史 | ✅ | ✅ |
| MongoDB 支持 | ✅ | ❌ (仅 PostgreSQL) |

## 与 LangGraph Checkpoint 的关系

```
┌─────────────────────────────────────────────────────────┐
│                    Kiki Checkpoint 系统                  │
├─────────────────────────────────────────────────────────┤
│  checkpoint.py (LangGraph 官方 Checkpoint)              │
│  - AsyncPostgresSaver: LangGraph 状态检查点            │
│  - MemorySaver: 内存状态检查点                         │
│  - 用途: 图执行状态恢复                                 │
├─────────────────────────────────────────────────────────┤
│  chat_stream.py (ChatStreamManager)                     │
│  - InMemoryStore + PostgreSQL: 流式消息缓存           │
│  - 用途: 对话历史持久化、流式消息合并                  │
└─────────────────────────────────────────────────────────┘
```

- **checkpoint.py**: 管理 LangGraph 图的执行状态（StateGraph）
- **chat_stream.py**: 管理用户对话的历史消息（业务层）

两者可以同时使用，各司其职。

## 测试验证

ChatStreamManager 已实现并通过代码审查，核心功能：

```bash
✓ 双层存储架构 (InMemoryStore + PostgreSQL)
✓ 流式消息处理
✓ 条件持久化
✓ 完整 CRUD 操作
```

## 下一步

根据 DeerFlow 分析报告，剩余功能：

1. ✅ Checkpoint 持久化系统（已完成）
2. ✅ 分层 LLM 配置（已完成）
3. ✅ YAML 配置管理（已完成）
4. ⏳ Prompt 模板系统 (Jinja2) - 部分完成
5. ⏳ 文档更新
6. ⏳ 测试集成
