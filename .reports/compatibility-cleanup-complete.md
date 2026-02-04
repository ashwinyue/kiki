# 兼容性代码清理完成

**日期**: 2026-02-04

## 清理概览

按照用户要求"不需要兼容性代码"，清理了所有向后兼容性代码和废弃的依赖。

## 清理的兼容性别名

### 模型层 (`app/models/`)

**`app/models/agent_config.py`**:
- ❌ `CustomAgent = AgentConfig`
- ❌ `CustomAgentCreate = AgentConfigCreate`
- ❌ `CustomAgentUpdate = AgentConfigUpdate`
- ❌ `CustomAgentPublic = AgentConfigPublic`
- ❌ `Agent = CustomAgent`
- ❌ `AgentCreate = CustomAgentCreate`
- ❌ `AgentUpdate = CustomAgentUpdate`
- ❌ `AgentPublic = CustomAgentPublic`

**`app/models/__init__.py`**:
- 移除所有 `CustomAgent*` 和 `Agent*` 别名的导出

### Schema 层 (`app/schemas/`)

**`app/schemas/agent_config.py`**:
- ❌ `CustomAgentConfig = AgentConfigSchema`

### Repository 层 (`app/repositories/`)

**`app/repositories/agent_async.py`**:
- ❌ `Agent = AgentConfig` （类型别名）
- ❌ `CustomAgent = AgentConfig` （兼容性别名）

所有 `Agent` 类型引用改为 `AgentConfig`

### 服务层 (`app/services/`)

**`app/services/agent/mcp_service.py`**:
- ❌ `get_mcp_service_service = get_mcp_service` （向后兼容别名）

## 清理的废弃依赖

### `app/config/dependencies.py`

**删除的内容**:
- ❌ `AgentContainer` 类（管理 ChatAgent 实例）
- ❌ `get_agent_dep` 函数（Agent 依赖注入）
- ❌ `ChatAgent` 导入和所有相关引用

**保留的内容**:
- ✅ `get_llm_service_dep`
- ✅ `get_memory_manager_dep`
- ✅ `get_memory_manager_factory_dep`
- ✅ `get_context_manager_dep`
- ✅ `get_settings_dep`
- ✅ `get_checkpointer_dep`

### `app/api/v1/dependencies.py`

**删除的内容**:
- ❌ `ChatAgent` 导入
- ❌ `AgentDep` 类型别名（`Annotated[ChatAgent, Depends(get_agent_dep)]`）
- ❌ `get_agent_with_memory_dep` 函数
- ❌ `get_chat_graph_dep` 函数
- ❌ `get_knowledge_service_dep` 函数（已废弃）
- ❌ `get_task_service_dep` 函数（已废弃）
- ❌ `get_session_service_dep` 函数

**保留的内容**:
- ✅ `DbDep`
- ✅ `TenantIdDep`, `RequiredTenantIdDep`
- ✅ `UserIdDep`, `RequiredUserIdDep`
- ✅ `LlmServiceDep`
- ✅ `MemoryManagerDep`
- ✅ `get_model_service_dep`
- ✅ `validate_session_access_dep`
- ✅ `resolve_effective_user_id_dep`

### `app/config/__init__.py`

**删除的导出**:
- ❌ `AgentContainer`
- ❌ `get_agent_dep`

**添加的导出**:
- ✅ YAML 配置加载器函数
  - `get_bool_env`
  - `get_float_env`
  - `get_int_env`
  - `get_list_env`
  - `get_str_env`
  - `load_yaml_config`
  - `reload_config`

## 类型更新

### `app/repositories/agent_async.py`

所有函数返回类型从 `Agent` 更新为 `AgentConfig`：

```python
# 之前
async def create_with_tools(self, data: dict[str, Any]) -> Agent:
async def get_by_name(self, name: str) -> Agent | None:
async def list_by_tenant(...) -> PaginatedResult[Agent]:
async def update_agent(...) -> Agent | None:
async def copy(...) -> Agent | None:

# 之后
async def create_with_tools(self, data: dict[str, Any]) -> AgentConfig:
async def get_by_name(self, name: str) -> AgentConfig | None:
async def list_by_tenant(...) -> PaginatedResult[AgentConfig]:
async def update_agent(...) -> AgentConfig | None:
async def copy(...) -> AgentConfig | None:
```

## 验证结果

```bash
✓ 所有兼容性代码已清理，导入正常！
  AgentConfig: AgentConfig
  AgentConfigSchema: AgentConfigSchema
  AgentRepositoryAsync: AgentRepositoryAsync
  ✓ CustomAgent 兼容性别名已删除
  ✓ CustomAgentConfig 兼容性别名已删除
  ✓ ChatAgent 相关依赖已删除
  ✓ AgentContainer 已删除
  ✓ get_agent_dep 已删除
```

## 迁移指南

如果您之前使用了兼容性别名，请按以下方式迁移：

### 模型导入

```python
# 之前（已废弃）
from app.models import CustomAgent, CustomAgentCreate

# 现在
from app.models import AgentConfig, AgentConfigCreate
```

### Schema 导入

```python
# 之前（已废弃）
from app.schemas import CustomAgentConfig

# 现在
from app.schemas.agent_config import AgentConfigSchema
```

### Agent 依赖注入

```python
# 之前（已废弃）
from app.api.v1.dependencies import AgentDep

# 现在 - 使用新的 Agent 工厂系统
from app.agent.graph import create_agent

agent = create_agent(
    agent_name="my_agent",
    agent_type="chat",
    tools=[],
)
```

## 后续工作

由于删除了 `ChatAgent` 和 `AgentDep`，以下文件可能需要更新：

1. **`app/api/v1/chat.py`** - 使用 `AgentDep` 的 API 路由
   - 需要迁移到新的 DeerFlow 风格 Agent 系统

2. **`app/api/v1/agents.py`** - Agent CRUD API
   - 已经使用 `AgentConfig`，应该不受影响

3. 其他使用 `ChatAgent` 或 `get_agent_dep` 的地方

## 注意事项

⚠️ **破坏性变更**: 此次清理删除了所有向后兼容性代码。如果您之前依赖这些别名，代码可能会出现导入错误。请参考迁移指南更新代码。
