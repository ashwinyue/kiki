# DeerFlow 功能实现总结

**日期**: 2026-02-04
**参考项目**: [ByteDance/deer-flow](https://github.com/bytedance/deer-flow)

## 实现概览

参考 DeerFlow 的设计，完成了生产级 Agent 框架的核心功能实现。

### 核心设计原则

- **简洁至上**: 所有 Agent 都是 `CompiledStateGraph`，无额外包装类
- **配置驱动**: 通过 YAML + 环境变量统一管理配置
- **分层架构**: 不同任务使用不同层级的 LLM
- **持久化优先**: 双层存储（InMemory + PostgreSQL）确保可靠性

---

## 功能实现状态

| 功能 | 状态 | 优先级 | 说明 |
|------|------|--------|------|
| Checkpoint 持久化 | ✅ 完成 | P0 | ChatStreamManager 双层存储 |
| 分层 LLM 配置 | ✅ 完成 | P0 | reasoning/basic/vision/code |
| YAML 配置管理 | ✅ 完成 | P1 | 环境变量替换，优先级管理 |
| Prompt 模板系统 | ✅ 完成 | P1 | Jinja2 多语言模板 |
| Agent 工厂 | ✅ 完成 | P0 | create_react_agent 统一接口 |
| 工具拦截器 | ✅ 完成 | P1 | Human-in-the-Loop |

---

## 详细实现

### 1. Checkpoint 持久化系统 ✅

**文件**: `app/agent/graph/chat_stream.py`

参考 DeerFlow 的 `ChatStreamManager` 设计：

```python
class ChatStreamManager:
    """流式对话消息管理器

    双层存储架构：
    1. InMemoryStore - 临时缓存流式消息块
    2. PostgreSQL - 持久化完整对话（finish_reason 触发时）
    """

    async def process_stream_message(
        self, thread_id: str, message: str, finish_reason: str
    ) -> bool:
        """处理流式消息"""
        # 1. 存储到 InMemoryStore
        # 2. finish_reason in ("stop", "interrupt") 时持久化
```

**核心特性**:
- 双层存储（InMemory + PostgreSQL）
- 消息块索引存储
- 条件持久化（finish_reason 触发）
- 完整 CRUD 操作

### 2. 分层 LLM 配置 ✅

**文件**: `app/agent/config/llm_config.py`

```python
LLMType = Literal["reasoning", "basic", "vision", "code"]

AGENT_LLM_MAP: dict[str, LLMType] = {
    "planner": "reasoning",     # DeepSeek Reasoner
    "coder": "code",            # Claude Sonnet
    "researcher": "basic",       # GPT-4o
    "supervisor": "reasoning",   # DeepSeek Reasoner
}

def get_llm_for_agent(agent_type: str) -> BaseChatModel:
    """根据 Agent 类型获取对应的 LLM"""
    llm_type = AGENT_LLM_MAP[agent_type]
    return get_llm_by_type(llm_type)
```

**YAML 配置**:
```yaml
REASONING_MODEL:
  model: "deepseek-reasoner"
  provider: "deepseek"
  temperature: 0.6
  max_tokens: 100000

CODE_MODEL:
  model: "claude-sonnet-4-20250514"
  provider: "anthropic"
  temperature: 0.3
```

### 3. YAML 配置管理 ✅

**文件**: `app/config/loader.py`

```python
def load_yaml_config(file_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件（带缓存）

    支持环境变量替换：
    - $VAR_NAME
    - ${VAR_NAME:-default}
    """

def _process_env_vars(config: Any) -> Any:
    """递归处理配置中的环境变量"""
```

**配置优先级**:
```
环境变量 > YAML 配置 > 默认值
```

**环境变量替换示例**:
```yaml
# conf.yaml
api_key: "$OPENAI_API_KEY"
base_url: "${API_BASE_URL:-https://api.openai.com}"
```

### 4. Prompt 模板系统 ✅

**文件**: `app/agent/prompts/template.py`

```python
def render_prompt(name: str, locale: str = "zh-CN", **variables) -> str:
    """渲染 Jinja2 模板

    支持：
    - 多语言（zh-CN, en-US, ja-JP）
    - 自动回退机制
    - 内置变量（locale, now, today, env）
    """

def create_langchain_prompt(name: str, **variables) -> ChatPromptTemplate:
    """创建 LangChain ChatPromptTemplate"""
```

**内置模板**:
- 通用：chat, chat_with_tools, router, supervisor
- 专门化角色：planner, researcher, analyst, coder, reporter

### 5. Agent 工厂 ✅

**文件**: `app/agent/graph/agent_factory.py`

```python
def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
) -> Any:
    """统一 Agent 创建工厂（返回 CompiledStateGraph）

    参考 DeerFlow 设计，所有 Agent 都是 CompiledStateGraph。
    """
    llm = get_llm_for_agent(agent_type)
    agent = create_react_agent(
        name=agent_name,
        model=llm,
        tools=tools,
        prompt=prompt_fn,
    )
    return agent
```

**设计原则**:
- 无额外包装类
- 统一创建接口
- 配置驱动（AGENT_LLM_MAP + prompt_template）

### 6. 工具拦截器 ✅

**文件**: `app/agent/tools/interceptor.py`

```python
class ToolInterceptor:
    """工具拦截器 - Human-in-the-Loop

    支持：
    - 拦截指定工具
    - 人工审批
    - 自动拒绝危险工具
    """

def wrap_tools_with_interceptor(
    tools: list[BaseTool],
    interceptor: ToolInterceptor,
) -> list[BaseTool]:
    """包装工具列表，添加拦截器"""
```

---

## 配置文件

### conf.example.yaml

完整的配置示例文件，包含：

```yaml
# 应用配置
APP_NAME: "Kiki Agent Framework"
DEBUG: false
ENVIRONMENT: "development"

# 数据库配置
DATABASE_URL: "postgresql+asyncpg://user:password@localhost:5432/kiki"

# 分层 LLM 配置
REASONING_MODEL:
  model: "deepseek-reasoner"
  provider: "deepseek"

BASIC_MODEL:
  model: "gpt-4o"
  provider: "openai"

CODE_MODEL:
  model: "claude-sonnet-4-20250514"
  provider: "anthropic"

# Agent LLM 映射
AGENT_LLM_MAPPING:
  coordinator: "basic"
  planner: "reasoning"
  coder: "code"
  researcher: "basic"

# Checkpoint 持久化
CHECKPOINT:
  enable_postgres: true
```

---

## 使用示例

### 1. 创建完整的研究团队

```python
from app.agent.graph import AGENT_REGISTRY, RESEARCH_TEAM, create_agent
from app.agent.workflow import run_agent_workflow

# 运行研究工作流
result = await run_agent_workflow(
    user_input="分析 2024 年 AI 行业发展趋势",
    max_step_num=5,
)
```

### 2. 创建单个 Agent

```python
from app.agent.graph import create_agent

# 创建 Planner
planner = create_agent(
    agent_name="planner",
    agent_type="planner",
    tools=[],
    prompt_template="planner",
)

# 创建 Coder
coder = create_agent(
    agent_name="coder",
    agent_type="coder",
    tools=[python_repl],
    prompt_template="coder",
)
```

### 3. 使用 Prompt 模板

```python
from app.agent.prompts.template import render_prompt

# 渲染 Planner 模板
prompt = render_prompt(
    "planner",
    goal="创建一个网站",
    context="使用 FastAPI",
    locale="zh-CN",
)
```

---

## 文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/config/loader.py` | YAML + 环境变量配置加载器 |
| `conf.example.yaml` | 配置示例文件 |
| `.reports/yaml-config-system-complete.md` | YAML 配置系统报告 |
| `.reports/checkpoint-persistence-complete.md` | Checkpoint 持久化报告 |
| `.reports/prompt-system-complete.md` | Prompt 模板系统报告 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app/config/settings.py` | 添加 YAML 配置加载支持 |
| `app/agent/config/llm_config.py` | 添加 YAML 配置加载，Agent-LLM 映射 |
| `app/config/__init__.py` | 导出 YAML 加载器函数 |
| `.gitignore` | 添加 `conf.yaml` |

### 已存在文件（已实现）

| 文件 | 说明 |
|------|------|
| `app/agent/graph/agent_factory.py` | Agent 工厂（DeerFlow 风格） |
| `app/agent/graph/chat_stream.py` | ChatStreamManager |
| `app/agent/prompts/template.py` | Prompt 模板系统 |
| `app/agent/tools/interceptor.py` | 工具拦截器 |

---

## DeerFlow 对比

| 功能 | DeerFlow | Kiki | 状态 |
|------|----------|------|------|
| **架构** |
| LangGraph StateGraph | ✅ | ✅ | 完成 |
| CompiledStateGraph | ✅ | ✅ | 完成 |
| 无包装类设计 | ✅ | ✅ | 完成 |
| **配置** |
| YAML 配置 | ✅ | ✅ | 完成 |
| 环境变量替换 | ✅ | ✅ | 完成 |
| 分层 LLM 配置 | ✅ | ✅ | 完成 |
| Agent-LLM 映射 | ✅ | ✅ | 完成 |
| **持久化** |
| InMemoryStore | ✅ | ✅ | 完成 |
| PostgreSQL 持久化 | ✅ | ✅ | 完成 |
| 消息块索引存储 | ✅ | ✅ | 完成 |
| 条件持久化 | ✅ | ✅ | 完成 |
| **Prompt** |
| Jinja2 模板 | ✅ | ✅ | 完成 |
| 多语言支持 | ✅ | ✅ | 完成 |
| 内置模板 | ✅ | ✅ | 完成 |
| **工具** |
| 工具拦截器 | ✅ | ✅ | 完成 |
| Human-in-the-Loop | ✅ | ✅ | 完成 |
| 工具注册系统 | ✅ | ✅ | 完成 |

---

## 下一步

1. **测试集成**: 编写单元测试和集成测试
2. **文档更新**: 更新用户文档和 API 文档
3. **示例应用**: 创建示例应用展示功能

---

## 参考资料

- DeerFlow 仓库: https://github.com/bytedance/deer-flow
- DeerFlow 分析报告: `.reports/deer-flow-analysis.md`
- 配置示例: `conf.example.yaml`
