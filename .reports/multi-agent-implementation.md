# Multi-Agent 架构实现总结

## 📋 实施概述

本次实现为 Kiki 添加了完整的 **Multi-Agent 架构支持**，参考 LangGraph 官方最佳实践，与 WeKnora99 的单 Agent 架构有本质区别。

---

## ✅ 已完成内容

### 1. 数据库层

**新增表：**
- `agent_executions` - Agent 执行记录表，追踪调用链和性能指标

**扩展表：**
- `sessions` - 添加 `graph_type`, `primary_agent_id`, `supervisor_config`
- `custom_agents` - 添加 `agent_role`, `parent_agent_id`, `allowed_workers`

**迁移文件：**
- `migrations/010_add_multi_agent_support.sql`
- `migrations/010_add_multi_agent_support.rollback.sql`

### 2. 模型层

**新增模型：**
- `AgentExecution` - Agent 执行记录模型
- `AgentType` - Agent 类型常量
- `ExecutionStatus` - 执行状态常量
- `AgentRole` - Agent 角色常量
- `GraphType` - 图类型常量

**更新模型：**
- `Session` - 扩展 Multi-Agent 支持
- `CustomAgent` - 扩展 Multi-Agent 角色

### 3. Graph 层

**新增文件：**
- `app/agent/graph/multi_agent.py` - Multi-Agent Graph Builder

**支持的模式：**
- ✅ **Supervisor Pattern** - 协调多个 worker agents
- ✅ **Router Pattern** - 意图路由
- 🔄 **Hierarchical Pattern** - 分层结构（架构已支持，待完善）

### 4. 测试脚本

- `app/agent/graph/test_multi_agent.py` - Multi-Agent 测试脚本

---

## 🏗️ 架构设计

### Supervisor Pattern（推荐）

```
        Main Graph (Supervisor)
               │
       ┌───────┼───────┐
       │       │       │
    RAG Agent Search Code
     (Worker)  (Worker) (Worker)
```

**核心特性：**
- Supervisor 节点使用 `Command` 对象路由到 workers
- 每个 worker 完成后返回 supervisor
- 支持追踪调用链（通过 `agent_executions` 表）

### 状态管理

```python
class MultiAgentState(ChatState):
    next_agent: str | None          # 下一个调用的 agent
    agent_outputs: dict[str, Any]   # 各 agent 的输出
    current_agent_role: str | None   # 当前 agent 角色
    parent_agent_id: str | None      # 父 agent ID
```

---

## 📊 与 WeKnora99 的关键差异

| 维度 | WeKnora99 | Kiki (Multi-Agent) |
|------|-----------|-------------------|
| **架构** | 单 Agent | **Multi-Agent (Supervisor)** |
| **路由** | CustomAgent.config | **Supervisor Graph + Send/Command** |
| **执行追踪** | Message.agent_steps | **AgentExecution 表（完整调用链）** |
| **Agent 通信** | 无 | **Send/Command 对象，支持嵌套** |
| **状态管理** | 单一 State | **分层独立 State** |
| **性能分析** | 基础日志 | **duration_ms, 调用链追踪** |

---

## 🔧 使用示例

### 创建 Supervisor Agent

```python
from app.agent.graph.multi_agent import build_multi_agent_graph

# 定义 Workers
workers = {
    "rag-agent": {
        "system_prompt": "你是知识库检索专家",
    },
    "search-agent": {
        "system_prompt": "你是网络搜索专家",
    },
    "code-agent": {
        "system_prompt": "你是代码执行专家",
    },
}

# 构建 Supervisor Graph
graph = await build_multi_agent_graph(
    graph_type="supervisor",
    workers=workers,
)

# 执行
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "搜索最新新闻"}]},
    {"configurable": {"thread_id": "session-123"}},
)

# 查看各 agent 的输出
agent_outputs = result.get("agent_outputs", {})
```

### 记录 Agent 执行

```python
from app.models.agent_execution import AgentExecution, ExecutionStatus
from datetime import UTC, datetime

# 创建执行记录
execution = AgentExecution(
    session_id="session-123",
    thread_id="session-123",
    agent_id="rag-agent",
    agent_type="worker",
    parent_execution_id=None,  # 顶层 agent
    input_data={"query": "用户问题"},
    status="running",
    started_at=datetime.now(UTC),
)

# 执行完成后更新
execution.status = "completed"
execution.output_data={"answer": "回答内容"}
execution.completed_at = datetime.now(UTC)
execution.duration_ms = 1500  # 1.5秒
```

---

## 🧪 测试

```bash
# 运行 Multi-Agent 测试
uv run python -m app.agent.graph.test_multi_agent

# 运行数据库迁移
psql -U your_user -d your_database -f migrations/010_add_multi_agent_support.sql

# 启动应用
uv run uvicorn app.main:app --reload
```

---

## 🎯 下一步（可选增强）

### P2 优先级

| # | 任务 | 说明 |
|---|------|------|
| 1 | **AgentExecutionRepository** | 数据访问层，查询调用链 |
| 2 | **Hierarchical Pattern** | 实现分层 agent 结构 |
| 3 | **Agent 性能监控** | 基于 duration_ms 的性能分析 |
| 4 | **Agent 调用链可视化** | 前端展示调用关系 |

### P3 优先级

| # | 任务 | 说明 |
|---|------|------|
| 1 | **动态 Worker 注册** | 运行时添加/移除 worker |
| 2 | **Agent 通信优化** | 减少跨 agent 数据传递开销 |
| 3 | **Agent 限流** | 防止 agent 过度调用 |

---

## 📚 参考资料

- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Supervisor Pattern](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Command & Send](https://langchain-ai.github.io/langgraph/reference/#langgraph.types.Command)
