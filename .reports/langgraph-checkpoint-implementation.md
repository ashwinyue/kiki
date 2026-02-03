# LangGraph Checkpoint 持久化 - 实现文档

## 概述

本次 P0 任务实现了 LangGraph AsyncPostgresSaver 的完整支持，包括：

1. ✅ 创建了三张 Checkpoint 表（`checks`, `checkpoints_blobs`, `checkpoint_writes`）
2. ✅ 实现了 AsyncPostgresSaver 的正确初始化和管理
3. ✅ 更新了图构建器以使用新的 checkpointer
4. ✅ 添加了应用生命周期的 checkpointer 管理

---

## 文件变更

### 1. 数据库迁移

**新增文件：**
- `migrations/009_add_langgraph_checkpoint_tables.sql` - 创建三张表
- `migrations/009_add_langgraph_checkpoint_tables.rollback.sql` - 回滚脚本

**表结构：**

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| `checks` | 存储检查点状态快照 | `thread_id`, `checkpoint_id`, `checkpoint` (JSONB) |
| `checkpoints_blobs` | 存储大对象（大消息、文件） | `thread_id`, `checkpoint_ns`, `blob` (BYTEA) |
| `checkpoint_writes` | 存储写入操作历史 | `thread_id`, `checkpoint_id`, `channel`, `value` (JSONB) |

**运行迁移：**
```bash
# 使用 psql 运行迁移
psql -U your_user -d your_database -f migrations/009_add_langgraph_checkpoint_tables.sql

# 或使用 Alembic（如果配置）
alembic upgrade head
```

---

### 2. Checkpoint 模块

**新增文件：**
- `app/agent/graph/checkpoint.py` - Checkpoint 初始化和管理

**核心函数：**

```python
# 获取 PostgreSQL checkpointer（单例）
checkpointer = await get_postgres_checkpointer()

# 获取 checkpointer（支持降级到 MemorySaver）
checkpointer = await get_checkpointer(use_postgres=True)

# 上下文管理器
async with checkpointer_context() as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# 列出检查点
checkpoints = await list_checkpoints(thread_id="session-123", limit=10)

# 获取检查点数量
count = await get_checkpoint_count(thread_id="session-123")

# 关闭 checkpointer（应用关闭时）
await close_postgres_checkpointer()
```

---

### 3. 图构建器更新

**修改文件：**
- `app/agent/graph/builder.py`

**新增参数：**
```python
async def compile_chat_graph(
    ...
    use_postgres_checkpointer: bool = True,  # 新增参数
) -> CompiledStateGraph:
    ...
```

**自动降级机制：**
- 如果 `use_postgres_checkpointer=True` 但 PostgreSQL 不可用
- 自动降级到 `MemorySaver` 并记录警告日志

---

### 4. 应用启动/关闭

**修改文件：**
- `app/main.py`

**启动时：**
- Checkpointer 采用懒加载，首次使用时初始化

**关闭时：**
- 在 `lifespan` 函数中添加 `close_checkpointer()` 任务
- 确保 checkpointer 在应用关闭前正确释放资源

---

## 使用示例

### 基础使用

```python
from app.agent.graph import compile_chat_graph

# 使用默认 PostgreSQL checkpointer
graph = await compile_chat_graph()

# 配置 thread_id（对应 sessions.id 或 threads.id）
config = {"configurable": {"thread_id": "session-123"}}

result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config,
)
```

### 检查点查询

```python
from app.agent.graph.checkpoint import (
    list_checkpoints,
    get_checkpoint_count,
)

# 列出检查点
checkpoints = await list_checkpoints(thread_id="session-123", limit=10)

# 获取检查点数量
count = await get_checkpoint_count(thread_id="session-123")
```

### 使用 MemorySaver（开发/测试）

```python
# 强制使用 MemorySaver
graph = await compile_chat_graph(use_postgres_checkpointer=False)

# 或者设置环境变量
# DATABASE_URL=sqlite:///./dev.db uv run uvicorn app.main:app --reload
```

---

## 配置要求

### 环境变量

```bash
# PostgreSQL 连接字符串（必需）
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/kiki

# 对于 AsyncPostgresSaver，会自动转换为：
# postgresql://user:pass@localhost:5432/kiki
```

### 依赖包

```bash
# 确保已安装（已在 pyproject.toml 中）
uv add langgraph-checkpoint-postgres
```

---

## 测试

### 运行测试脚本

```bash
# 测试 Checkpoint 基础功能
uv run python -m app.agent.graph.test_checkpoint

# 预期输出：
# ✅ checkpointer_initialized
# ✅ graph_compiled_success
# ✅ graph_executed_success
# ✅ checkpoint_count_retrieved
# ✅ persistence_check
```

### 手动验证

```python
import asyncio
from app.agent.graph import compile_chat_graph

async def main():
    graph = await compile_chat_graph()

    # 第一次调用
    result1 = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "1+1=?"}]},
        {"configurable": {"thread_id": "test-001"}}
    )
    print(f"第一次调用: {len(result1['messages'])} 条消息")

    # 第二次调用（会从 checkpoint 恢复）
    result2 = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "再加2等于多少？"}]},
        {"configurable": {"thread_id": "test-001"}}
    )
    print(f"第二次调用: {len(result2['messages'])} 条消息（应该包含历史）")

asyncio.run(main())
```

---

## 故障排查

### 问题 1：PostgreSQL checkpointer 初始化失败

**错误信息：**
```
RuntimeError: Failed to initialize PostgreSQL checkpointer: ...
```

**解决方案：**
1. 检查 `DATABASE_URL` 是否正确
2. 确保数据库是 PostgreSQL（不是 SQLite）
3. 检查 `langgraph-checkpoint-postgres` 是否已安装
4. 查看日志中的详细错误信息

### 问题 2：表未创建

**错误信息：**
```
relation "checks" does not exist
```

**解决方案：**
```bash
# 运行迁移脚本
psql -U your_user -d your_database -f migrations/009_add_langgraph_checkpoint_tables.sql
```

### 问题 3：降级到 MemorySaver

**警告信息：**
```
postgres_checkpointer_fallback, fallback="memory_saver"
```

**这是正常的降级行为：**
- 开发环境使用 SQLite → 自动降级
- PostgreSQL 连接失败 → 自动降级

**如果想强制使用 PostgreSQL：**
- 确保使用 PostgreSQL 数据库
- 检查连接字符串格式
- 查看日志中的详细错误

---

## 最佳实践

### 1. Thread ID 管理

```python
# 推荐：使用 Session ID 作为 thread_id
thread_id = session.id  # 对应 sessions 表

# 或使用 Thread ID
thread_id = thread.id   # 对应 threads 表
```

### 2. Checkpoint 清理

```sql
-- 定期清理旧检查点（30 天前）
SELECT cleanup_old_checkpoints(30);
```

### 3. 监控检查点数量

```python
from app.agent.graph.checkpoint import get_checkpoint_count

# 定期检查，防止检查点过多
count = await get_checkpoint_count(thread_id="session-123")
if count > 1000:
    # 触发清理或告警
    logger.warning("too_many_checkpoints", thread_id="session-123", count=count)
```

---

## LangGraph 最佳实践对齐

| 功能 | 实现状态 | 说明 |
|------|----------|------|
| **Thread 表** | ✅ 已有 | 对应 LangGraph 的 `thread_id` |
| **Checkpoint 表** | ✅ 本次新增 | 三张表：`checks`, `checkpoints_blobs`, `checkpoint_writes` |
| **AsyncPostgresSaver** | ✅ 本次新增 | 完整支持，带自动降级 |
| **Memory 表** | ✅ 已有 | 用于跨会话的长期记忆（LangGraph Store） |
| **应用生命周期管理** | ✅ 本次新增 | 启动/关闭时正确管理 checkpointer |

---

## 下一步（P1 优先级）

1. ⚠️ 简化 Session 表，将配置移至 CustomAgent
2. ⚠️ 完善 CustomAgent.config 结构
3. ⚠️ 实现内置 Agent（quick-answer, smart-reasoning）
