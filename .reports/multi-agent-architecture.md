# Kiki Multi-Agent 架构设计（参考 LangGraph 最佳实践）

## 一、LangGraph Multi-Agent 模式

### 1.1 官方推荐模式

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| **Supervisor** | 需要协调多个专家 Agent | Central coordinator routes to workers |
| **Router** | 简单场景，按意图路由 | Intent classification → single agent |
| **Swarm** | 并行处理多个子任务 | Agents work in parallel, merge results |
| **Hierarchical** | 复杂任务分解 | Layered agent structure |

### 1.2 Kiki 推荐方案

**采用 Supervisor Pattern + Router Pattern 混合架构**

```
                    ┌─────────────────┐
                    │   Main Graph    │
                    │  (Supervisor)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌───▼────┐  ┌─────▼─────┐
        │  RAG Agent │  │ Search│  │  Code     │
        │  (Worker)  │  │ Agent │  │  Agent    │
        └───────────┘  └────────┘  └───────────┘
```

---

## 二、数据库表设计（Multi-Agent 专用）

### 2.1 新增表：`agent_executions`

记录 Agent 调用链路，支持：
- 追踪哪个 agent 调用了哪个 agent
- 记录每个 agent 的执行时间和状态
- 支持调试和性能分析

```sql
CREATE TABLE agent_executions (
    id UUID PRIMARY KEY,
    session_id VARCHAR(36) REFERENCES sessions(id),
    thread_id VARCHAR(36) REFERENCES threads(id),

    -- Agent 调用信息
    agent_id VARCHAR(64) NOT NULL,           -- 被调用的 Agent ID
    agent_type VARCHAR(50) NOT NULL,         -- agent 类型
    parent_execution_id UUID,                 -- 父执行 ID（形成调用链）

    -- 执行信息
    input_data JSONB,                        -- 输入数据
    output_data JSONB,                       -- 输出数据
    status VARCHAR(20) DEFAULT 'pending',   -- pending/running/completed/failed
    error_message TEXT,

    -- 性能指标
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,

    -- 元数据
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_executions_session ON agent_executions(session_id);
CREATE INDEX idx_agent_executions_thread ON agent_executions(thread_id);
CREATE INDEX idx_agent_executions_parent ON agent_executions(parent_execution_id);
```

### 2.2 修改：`sessions` 表

添加 `graph_type` 字段，支持不同的 Agent 图类型：

| 字段 | 类型 | 说明 |
|------|------|------|
| `graph_type` | VARCHAR(50) | `supervisor`, `router`, `single`, `hierarchical` |
| `primary_agent_id` | VARCHAR(64) | 主要 Agent ID（用于 single 模式） |
| `supervisor_config` | JSONB | Supervisor 配置（用于 supervisor 模式） |

### 2.3 修改：`custom_agents` 表

添加 Multi-Agent 相关字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_role` | VARCHAR(50) | `supervisor`, `worker`, `router`, `leaf` |
| `parent_agent_id` | VARCHAR(64) | 父 Agent ID（用于 hierarchical） |
| `allowed_workers` | JSONB | 允许调用的 worker agent ID 列表 |

---

## 三、Schema 定义

### 3.1 AgentExecution 模型

```python
class AgentExecutionBase(SQLModel):
    """Agent 执行基础模型"""

    session_id: str
    thread_id: str
    agent_id: str
    agent_type: str  # supervisor, rag_agent, search_agent, code_agent


class AgentExecution(AgentExecutionBase, table=True):
    """Agent 执行记录表"""

    __tablename__ = "agent_executions"

    id: str = Field(default=None, primary_key=True)

    # 调用链
    parent_execution_id: str | None = None

    # 执行数据
    input_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    status: str = Field(default="pending")
    error_message: str | None = None

    # 性能指标
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None

    # 元数据
    metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

### 3.2 Session 扩展（Multi-Agent 支持）

```python
class Session(SessionBase, table=True):
    """会话表（Multi-Agent 支持）"""

    # ... 现有字段 ...

    # Multi-Agent 配置
    graph_type: str = Field(
        default="single",
        description="图类型: single, supervisor, router, hierarchical"
    )
    primary_agent_id: str | None = Field(
        default=None,
        max_length=64,
        description="主要 Agent ID（single 模式使用）"
    )
    supervisor_config: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Supervisor 配置（supervisor 模式使用）"
    )
```

### 3.3 CustomAgent 扩展（Multi-Agent 角色）

```python
class CustomAgent(CustomAgentBase, table=True):
    """自定义 Agent 表（Multi-Agent 支持）"""

    # ... 现有字段 ...

    # Multi-Agent 角色
    agent_role: str = Field(
        default="leaf",
        description="Agent 角色: supervisor, worker, router, leaf"
    )
    parent_agent_id: str | None = Field(
        default=None,
        max_length=64,
        description="父 Agent ID（hierarchical 模式）"
    )
    allowed_workers: list[str] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="允许调用的 worker agent ID 列表"
    )
```

---

## 四、LangGraph Multi-Agent 实现模式

### 4.1 Supervisor Pattern（推荐）

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command

# 定义 Agent 状态
class SupervisorState(TypedDict):
    messages: list[BaseMessage]
    next: str  # 下一个要调用的 agent
    agent_outputs: dict[str, Any]  # 各 agent 的输出

# Supervisor 节点
def supervisor_node(state: SupervisorState) -> Command[Literal["agent_a", "agent_b", "agent_c"]]:
    """决定调用哪个 agent"""
    # 基于 LLM 决策或规则路由
    return Command(goto=next_agent)

# Worker Agent 节点
def agent_a_node(state: SupervisorState) -> SupervisorState:
    """Agent A 的逻辑"""
    result = do_agent_a_work()
    return {"agent_outputs": {"agent_a": result}}

# 构建图
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("agent_a", agent_a_node)
builder.add_node("agent_b", agent_b_node)
builder.add_node("agent_c", agent_c_node)

# 条件边：supervisor -> workers
builder.add_conditional_edges("supervisor", lambda s: s["next"])

# 每个 worker 完成后回到 supervisor
builder.add_edge("agent_a", "supervisor")
builder.add_edge("agent_b", "supervisor")
builder.add_edge("agent_c", "supervisor")
```

### 4.2 记录 Agent 调用

```python
from app.models.agent_execution import AgentExecution

async def execute_agent_with_tracking(
    agent_id: str,
    session_id: str,
    input_data: dict,
    parent_execution_id: str | None = None,
):
    """执行 agent 并记录调用链"""

    # 创建执行记录
    execution = AgentExecution(
        session_id=session_id,
        thread_id=session_id,  # 可以复用 session_id
        agent_id=agent_id,
        agent_type="worker",
        parent_execution_id=parent_execution_id,
        input_data=input_data,
        status="running",
        started_at=datetime.now(UTC),
    )

    # 保存到数据库
    async with session_scope() as session:
        session.add(execution)
        await session.commit()

    # 执行 agent
    try:
        result = await run_agent(agent_id, input_data)

        # 更新执行记录
        execution.status = "completed"
        execution.output_data = result
        execution.completed_at = datetime.now(UTC)
        execution.duration_ms = int(
            (execution.completed_at - execution.started_at).total_seconds() * 1000
        )

    except Exception as e:
        execution.status = "failed"
        execution.error_message = str(e)
        execution.completed_at = datetime.now(UTC)

    # 保存更新
    async with session_scope() as session:
        session.add(execution)
        await session.commit()

    return execution
```

---

## 五、实施步骤

### Phase 1: 数据库层
- [ ] 创建 `agent_executions` 表迁移
- [ ] 扩展 `sessions` 表（添加 `graph_type` 等）
- [ ] 扩展 `custom_agents` 表（添加 `agent_role` 等）

### Phase 2: 模型层
- [ ] 创建 `AgentExecution` SQLModel
- [ ] 创建 `AgentExecutionRepository`
- [ ] 创建 `MultiAgentGraphBuilder`

### Phase 3: Agent 层
- [ ] 实现 `SupervisorAgent`
- [ ] 实现 `WorkerAgent` 基类
- [ ] 实现内置 Workers（RAG, Search, Code）

### Phase 4: API 层
- [ ] 添加 Multi-Agent chat API
- [ ] 添加 Agent 调用链查询 API
- [ ] 添加 Agent 性能统计 API

---

## 六、与 WeKnora99 的差异

| 特性 | WeKnora99 | Kiki (Multi-Agent) |
|------|-----------|-------------------|
| **架构** | 单 Agent | Multi-Agent (Supervisor Pattern) |
| **路由** | CustomAgent.config | Supervisor Graph + Send/Command |
| **执行追踪** | Message.agent_steps | AgentExecution 表 |
| **Agent 通信** | 无 | Send 对象，支持嵌套调用 |
| **状态管理** | 单一 State | 每层独立的 State |

---

## 七、配置示例

### Supervisor Agent 配置

```python
supervisor_agent = CustomAgent(
    id="supervisor-main",
    name="主协调器",
    agent_role="supervisor",
    config=CustomAgentConfig(
        agent_mode="smart-reasoning",
        system_prompt="你是任务协调器，负责将用户问题分发给最合适的专家 agent。",
        tool=ToolConfig(
            allowed_tools=["route_to_agent"],
            allowed_workers=["rag-agent", "search-agent", "code-agent"],
        ),
    ),
)
```

### Worker Agent 配置

```python
rag_agent = CustomAgent(
    id="rag-agent",
    name="知识库专家",
    agent_role="worker",
    config=CustomAgentConfig(
        agent_mode="quick-answer",
        system_prompt="你是知识库检索专家，负责回答基于文档的问题。",
        knowledge_base=KnowledgeBaseConfig(
            kb_selection_mode="all",
        ),
    ),
)
```

---

是否按照这个 Multi-Agent 架构继续实现？
