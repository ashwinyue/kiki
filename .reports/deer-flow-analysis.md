# DeerFlow 项目分析报告

> 分析日期: 2025-02-03
> 分析目标: 提取 DeerFlow 项目中可引入 Kiki 的核心模块和设计模式

## 一、项目概览

### 1.1 项目定位

**DeerFlow** (**D**eep **E**xploration and **E**fficient **R**esearch **Flow**) 是字节跳动开源的多 Agent 深度研究框架，基于 LangGraph 构建生产级的 Agent 应用。

- **仓库**: https://github.com/bytedance/deer-flow
- **技术栈**: Python 3.12+, LangGraph, LangChain, FastAPI
- **应用场景**: 深度研究、报告生成、Podcast/PTT 内容创作

### 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph StateGraph                     │
├─────────────────────────────────────────────────────────────────┤
│  Coordinator → Background Investigator → Planner → Research Team │
│                                                          ↓        │
│  Human Feedback ←────────────────────────────────── Reporter    │
│                                                                  │
│  Research Team:                                                  │
│    - Researcher (Web Search, Crawl, MCP Tools)                   │
│    - Analyst (Data Analysis)                                    │
│    - Coder (Python REPL)                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈对比

| 组件 | DeerFlow | Kiki | 重叠度 |
|------|----------|------|--------|
| Web 框架 | FastAPI + Uvicorn | FastAPI + Uvicorn | ✅ 100% |
| 包管理 | uv | uv | ✅ 100% |
| Agent 框架 | LangGraph + LangChain | LangGraph + LangChain | ✅ 100% |
| 数据库 | PostgreSQL + MongoDB | PostgreSQL | ✅ 80% |
| Checkpoint | PostgreSQL/MongoDB + InMemory | (待实现) | ❌ 0% |
| Prompt 系统 | Jinja2 模板 | 字符串模板 | ⚠️ 50% |

---

## 二、可引入核心模块详解

### 2.1 Checkpoint 持久化系统 ⭐⭐⭐⭐⭐

**文件**: `src/graph/checkpoint.py`

#### 核心特性

```python
class ChatStreamManager:
    """
    双层存储架构：
    1. InMemoryStore - 临时缓存流式消息
    2. MongoDB/PostgreSQL - 完整对话持久化
    """
```

#### 关键设计

| 特性 | 实现方式 | 引入价值 |
|------|----------|----------|
| **双层存储** | 内存缓存 + 数据库持久化 | 性能优化 |
| **流式合并** | 按 chunk 索引存储，自动合并 | 支持流式输出 |
| **条件持久化** | finish_reason in ("stop", "interrupt") | 减少无效写入 |
| **多后端支持** | MongoDB + PostgreSQL | 灵活部署 |

#### 实现细节

```python
# 消息处理流程
def process_stream_message(thread_id, message, finish_reason):
    # 1. 创建命名空间
    store_namespace = ("messages", thread_id)

    # 2. 获取/初始化游标
    cursor = self.store.get(store_namespace, "cursor")
    current_index = cursor.value.get("index", 0) + 1

    # 3. 存储消息块
    self.store.put(store_namespace, f"chunk_{current_index}", message)

    # 4. 判断是否持久化
    if finish_reason in ("stop", "interrupt"):
        return self._persist_complete_conversation(
            thread_id, store_namespace, current_index
        )

# PostgreSQL 表结构
CREATE TABLE chat_streams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    messages JSONB NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

#### 引入建议

```python
# app/agent/graph/checkpoint.py (新建)

from langgraph.store.memory import InMemoryStore
import psycopg
from typing import Optional, List
import json
from datetime import datetime
import uuid

class ChatStreamManager:
    """Kiki 版本的 Checkpoint 管理器"""

    def __init__(self, checkpoint_saver: bool = False, db_uri: Optional[str] = None):
        self.store = InMemoryStore()
        self.checkpoint_saver = checkpoint_saver
        self.db_uri = db_uri or settings.LANGGRAPH_CHECKPOINT_DB_URL
        self.postgres_conn = None

        if self.checkpoint_saver:
            self._init_postgresql()

    def _init_postgresql(self):
        """初始化 PostgreSQL 连接"""
        self.postgres_conn = psycopg.connect(
            self.db_uri,
            row_factory=dict_row
        )
        self._create_chat_streams_table()

    def _create_chat_streams_table(self):
        """创建聊天流表"""
        # ... 实现同 deer-flow ...

    def process_stream_message(
        self, thread_id: str, message: str, finish_reason: str
    ) -> bool:
        """处理流式消息"""
        # ... 实现同 deer-flow ...
```

---

### 2.2 LLM 服务抽象层 ⭐⭐⭐⭐

**文件**: `src/llms/llm.py`

#### 分层 LLM 架构

```python
LLMType = Literal["reasoning", "basic", "vision", "code"]

# 配置映射
LLM_TYPE_CONFIG_KEYS = {
    "reasoning": "REASONING_MODEL",
    "basic": "BASIC_MODEL",
    "vision": "VISION_MODEL",
    "code": "CODE_MODEL",
}
```

#### 核心特性

| 特性 | 实现方式 | 引入价值 |
|------|----------|----------|
| **分层配置** | reasoning/basic/vision/code | 场景化模型选择 |
| **配置合并** | YAML + 环境变量 | 灵活部署 |
| **实例缓存** | `_llm_cache` dict | 性能优化 |
| **Token 推断** | 根据模型名称自动推断 | 防止溢出错误 |

#### 配置优先级

```python
# 1. 环境变量优先
env_conf = {
    f"{key[len(prefix):]}": value
    for key, value in os.environ.items()
    if key.startswith(f"{llm_type.upper()}_MODEL__")
}

# 2. YAML 配置次之
yaml_conf = conf.get(config_key, {})

# 3. 合并配置
merged_conf = {**yaml_conf, **env_conf}

# 4. 过滤无效参数
allowed_keys = {"model", "api_key", "base_url", ...}
filtered_conf = {
    k: v for k, v in merged_conf.items()
    if k.lower() in {ak.lower() for ak in allowed_keys}
}
```

#### Token Limit 推断

```python
def _infer_token_limit_from_model(model_name: str) -> int:
    """根据模型名称推断 Token 限制"""
    defaults = {
        "gpt-4o": 120000,
        "gpt-4-turbo": 120000,
        "claude-3": 180000,
        "gemini-2": 180000,
        "doubao": 200000,
        "deepseek": 100000,
        "default": 100000,
    }

    model_lower = model_name.lower()
    for key, limit in defaults.items():
        if key in model_lower:
            return limit
    return defaults["default"]
```

#### 引入建议

```python
# app/llm/service.py (重构)

from enum import Literal
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

LLMType = Literal["reasoning", "basic", "vision", "code"]

class LLMService:
    """统一的 LLM 服务管理"""

    _cache: dict[LLMType, BaseChatModel] = {}

    @classmethod
    def get_llm(cls, llm_type: LLMType) -> BaseChatModel:
        """获取 LLM 实例（带缓存）"""
        if llm_type in cls._cache:
            return cls._cache[llm_type]

        # 加载配置
        config = cls._load_config(llm_type)

        # 创建实例
        llm = ChatOpenAI(**config)
        cls._cache[llm_type] = llm
        return llm

    @classmethod
    def _load_config(cls, llm_type: LLMType) -> dict:
        """加载 LLM 配置"""
        # 1. 从配置文件加载
        yaml_conf = settings.get(f"{llm_type.upper()}_MODEL", {})

        # 2. 从环境变量加载
        prefix = f"{llm_type.upper()}_MODEL__"
        env_conf = {
            key[len(prefix):].lower(): value
            for key, value in os.environ.items()
            if key.startswith(prefix)
        }

        # 3. 合并配置
        return {**yaml_conf, **env_conf}
```

---

### 2.3 Prompt 模板系统 ⭐⭐⭐⭐

**文件**: `src/prompts/template.py`

#### Jinja2 模板架构

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

# 初始化 Jinja2 环境
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
```

#### 多语言支持

```python
def get_prompt_template(prompt_name: str, locale: str = "en-US") -> str:
    """
    模板文件命名规则：
    - researcher.md         (默认英文)
    - researcher.zh_CN.md   (中文)
    - researcher.ja_JP.md   (日文)
    """
    normalized_locale = locale.replace("-", "_")

    # 1. 尝试本地化模板
    try:
        template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
        return template.render()
    except TemplateNotFound:
        # 2. 降级到英文模板
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
```

#### 变量注入

```python
def apply_prompt_template(
    prompt_name: str,
    state: AgentState,
    configurable: Configuration = None,
    locale: str = "en-US"
) -> list:
    """
    自动注入变量：
    - CURRENT_TIME: 当前时间
    - state: Agent 状态
    - configurable: 配置对象
    """
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    template = env.get_template(f"{prompt_name}.{normalized_locale}.md")
    system_prompt = template.render(**state_vars)

    return [{"role": "system", "content": system_prompt}] + state["messages"]
```

#### 模板示例

```jinja2
<!-- prompts/researcher.zh_CN.md -->
# 研究员 Agent

你是一个专业的研究员，负责进行深入的信息收集和分析。

## 当前任务
{{ research_topic }}

## 已有资源
{% for resource in resources %}
- {{ resource.title }}: {{ resource.url }}
{% endfor %}

## 搜索策略
1. 使用 {{ max_search_results }} 个搜索结果
2. 关注 {{ include_domains }} 域名
3. 排除 {{ exclude_domains }} 域名

## 当前时间
{{ CURRENT_TIME }}
```

#### 引入建议

```python
# app/agent/prompts/template.py (新建)

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from datetime import datetime

# 初始化环境
TEMPLATES_DIR = Path(__file__).parent
env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_prompt(
    template_name: str,
    state: dict,
    locale: str = "zh-CN"
) -> str:
    """渲染 Prompt 模板"""
    # 1. 准备变量
    variables = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }

    # 2. 加载模板
    normalized_locale = locale.replace("-", "_")
    try:
        template = env.get_template(f"{template_name}.{normalized_locale}.md")
    except:
        template = env.get_template(f"{template_name}.md")

    # 3. 渲染
    return template.render(**variables)

# 目录结构
# app/agent/prompts/
# ├── __init__.py
# ├── template.py
# ├── researcher.md
# ├── researcher.zh_CN.md
# ├── coder.md
# └── coder.zh_CN.md
```

---

### 2.4 工具拦截器系统 ⭐⭐⭐⭐

**文件**: `src/agents/tool_interceptor.py`

#### 核心机制

```python
class ToolInterceptor:
    """工具执行拦截器 - 实现 Human-in-the-Loop"""

    def __init__(self, interrupt_before_tools: Optional[List[str]] = None):
        self.interrupt_before_tools = interrupt_before_tools or []

    def should_interrupt(self, tool_name: str) -> bool:
        """判断是否需要拦截"""
        return tool_name in self.interrupt_before_tools
```

#### 工具包装

```python
@staticmethod
def wrap_tool(tool: BaseTool, interceptor: "ToolInterceptor") -> BaseTool:
    """包装工具，添加拦截逻辑"""
    original_func = tool.func

    def intercepted_func(*args, **kwargs):
        tool_name = tool.name

        # 1. 检查是否需要拦截
        if interceptor.should_interrupt(tool_name):
            # 2. 触发中断，等待用户反馈
            feedback = interrupt(
                f"即将执行工具: '{tool_name}'\n\n"
                f"输入参数:\n{args[0]}\n\n"
                f"是否批准执行？"
            )

            # 3. 解析审批结果
            if not ToolInterceptor._parse_approval(feedback):
                return {
                    "error": "工具执行被用户拒绝",
                    "tool": tool_name,
                    "status": "rejected",
                }

        # 4. 执行原始工具
        return original_func(*args, **kwargs)

    # 5. 替换工具函数
    object.__setattr__(tool, "func", intercepted_func)
    return tool
```

#### 审批关键词

```python
def _parse_approval(feedback: str) -> bool:
    """解析用户反馈是否为批准"""
    approval_keywords = [
        "approved", "approve", "yes", "proceed",
        "continue", "ok", "okay", "accepted", "accept",
    ]

    feedback_lower = feedback.lower().strip()
    return any(keyword in feedback_lower for keyword in approval_keywords)
```

#### 引入建议

```python
# app/agent/tools/interceptor.py (增强)

from typing import List, Optional, Any
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

class ToolInterceptor:
    """工具执行拦截器"""

    def __init__(
        self,
        interrupt_before_tools: Optional[List[str]] = None,
        require_approval: bool = True,
    ):
        self.interrupt_before_tools = interrupt_before_tools or []
        self.require_approval = require_approval

    def wrap_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """批量包装工具"""
        if not self.interrupt_before_tools:
            return tools

        wrapped_tools = []
        for tool in tools:
            if tool.name in self.interrupt_before_tools:
                wrapped_tools.append(self._wrap_tool(tool))
            else:
                wrapped_tools.append(tool)
        return wrapped_tools

    def _wrap_tool(self, tool: BaseTool) -> BaseTool:
        """包装单个工具"""
        original_func = tool.func

        def intercepted_func(*args, **kwargs):
            if self.require_approval:
                feedback = interrupt(
                    f"即将执行工具: {tool.name}\n"
                    f"参数: {args[0] if args else kwargs}\n"
                    f"是否批准？"
                )
                if not self._is_approved(feedback):
                    return {"status": "rejected", "tool": tool.name}

            return original_func(*args, **kwargs)

        object.__setattr__(tool, "func", intercepted_func)
        return tool

    @staticmethod
    def _is_approved(feedback: str) -> bool:
        """判断是否批准"""
        keywords = ["yes", "ok", "approved", "proceed", "continue"]
        return any(kw in feedback.lower() for kw in keywords)

# 使用示例
# 在 Agent 创建时
interceptor = ToolInterceptor(
    interrupt_before_tools=["python_repl", "web_search"]
)
tools = interceptor.wrap_tools(tools)
```

---

### 2.5 配置管理系统 ⭐⭐⭐⭐

**文件**: `src/config/loader.py`

#### 统一配置加载

```python
import yaml
import os
from typing import Any, Dict

_config_cache: Dict[str, Dict[str, Any]] = {}

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    特性：
    1. 文件不存在返回空字典
    2. 配置缓存机制
    3. 环境变量替换
    """
    # 1. 文件不存在
    if not os.path.exists(file_path):
        return {}

    # 2. 检查缓存
    if file_path in _config_cache:
        return _config_cache[file_path]

    # 3. 加载 YAML
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # 4. 递归处理环境变量
    processed_config = process_dict(config)

    # 5. 存入缓存
    _config_cache[file_path] = processed_config
    return processed_config
```

#### 环境变量替换

```python
def replace_env_vars(value: str) -> str:
    """
    支持 $VAR_NAME 语法

    示例:
    api_key: $OPENAI_API_KEY
    base_url: $API_BASE_URL
    """
    if not isinstance(value, str):
        return value
    if value.startswith("$"):
        env_var = value[1:]
        return os.getenv(env_var, env_var)
    return value

def process_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """递归处理字典"""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = process_dict(value)
        elif isinstance(value, str):
            result[key] = replace_env_vars(value)
        else:
            result[key] = value
    return result
```

#### 类型安全辅助函数

```python
def get_bool_env(name: str, default: bool = False) -> bool:
    """获取布尔环境变量"""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def get_str_env(name: str, default: str = "") -> str:
    """获取字符串环境变量"""
    val = os.getenv(name)
    return default if val is None else val.strip()

def get_int_env(name: str, default: int = 0) -> int:
    """获取整数环境变量"""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default
```

#### 配置示例

```yaml
# conf.yaml
BASIC_MODEL:
  model: gpt-4o
  api_key: $OPENAI_API_KEY
  base_url: $OPENAI_BASE_URL
  max_retries: 3
  temperature: 0.7

REASONING_MODEL:
  model: deepseek-reasoner
  api_key: $DEEPSEEK_API_KEY
  base_url: https://api.deepseek.com

SEARCH_ENGINE:
  provider: tavily
  include_domains: []
  exclude_domains: ["spam.com"]
  search_depth: advanced
```

#### 引入建议

```python
# app/config/loader.py (新建)

import yaml
import os
from functools import lru_cache
from typing import Any, Dict

@lru_cache(maxsize=32)
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件（带缓存）"""
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return _process_env_vars(config)

def _process_env_vars(config: Any) -> Any:
    """递归处理环境变量"""
    if isinstance(config, dict):
        return {k: _process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_process_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("$"):
        return os.getenv(config[1:], config)
    return config

# app/config/dependencies.py (增强)

from .loader import load_yaml_config

def get_settings() -> Settings:
    """获取设置（从 YAML 文件）"""
    config = load_yaml_config("conf.yaml")

    return Settings(
        # 基础配置
        app_name=config.get("APP_NAME", "Kiki"),
        debug=config.get("DEBUG", False),

        # LLM 配置
        openai_api_key=config.get("OPENAI_API_KEY", ""),
        openai_base_url=config.get("OPENAI_BASE_URL"),

        # 分层 LLM
        basic_model=config.get("BASIC_MODEL", {}),
        reasoning_model=config.get("REASONING_MODEL", {}),

        # 其他配置...
    )
```

---

### 2.6 多 Agent 架构模式 ⭐⭐⭐⭐

**文件**: `src/graph/builder.py`, `src/agents/agents.py`

#### Agent 创建工厂

```python
def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list,
    prompt_template: str,
    pre_model_hook: callable = None,
    interrupt_before_tools: Optional[List[str]] = None,
):
    """
    统一的 Agent 创建工厂

    参数:
        agent_name: Agent 名称
        agent_type: Agent 类型（决定 LLM 类型）
        tools: 工具列表
        prompt_template: Prompt 模板名称
        pre_model_hook: 模型调用前钩子
        interrupt_before_tools: 需要拦截的工具列表
    """
    # 1. 工具处理
    processed_tools = tools
    if interrupt_before_tools:
        processed_tools = wrap_tools_with_interceptor(
            tools, interrupt_before_tools
        )

    # 2. LLM 选择
    llm_type = AGENT_LLM_MAP.get(agent_type, "basic")
    llm = get_llm_by_type(llm_type)

    # 3. 创建 ReAct Agent
    agent = create_react_agent(
        name=agent_name,
        model=llm,
        tools=processed_tools,
        prompt=lambda state: apply_prompt_template(
            prompt_template, state,
            locale=state.get("locale", "en-US")
        ),
        pre_model_hook=pre_model_hook,
    )

    return agent
```

#### Agent-LLM 映射

```python
# src/config/agents.py
AGENT_LLM_MAP = {
    "coordinator": "basic",
    "planner": "reasoning",
    "researcher": "basic",
    "analyst": "basic",
    "coder": "code",
    "reporter": "basic",
}
```

#### 条件边路由

```python
def continue_to_running_research_team(state: State):
    """
    根据当前计划状态路由到不同的 Agent

    路由逻辑：
    1. 没有计划 → planner
    2. 所有步骤完成 → planner
    3. 找到未完成的步骤：
       - RESEARCH → researcher
       - ANALYSIS → analyst
       - PROCESSING → coder
    """
    current_plan = state.get("current_plan")

    # 没有计划或所有步骤完成
    if not current_plan or all(step.execution_res for step in current_plan.steps):
        return "planner"

    # 找到第一个未完成的步骤
    for step in current_plan.steps:
        if not step.execution_res:
            if step.step_type == StepType.RESEARCH:
                return "researcher"
            elif step.step_type == StepType.ANALYSIS:
                return "analyst"
            elif step.step_type == StepType.PROCESSING:
                return "coder"

    return "planner"
```

#### 图构建模式

```python
def _build_base_graph():
    """构建基础状态图"""
    builder = StateGraph(State)

    # 添加节点
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)

    # 添加边
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "analyst", "coder"],
    )
    builder.add_edge("reporter", END)

    return builder
```

#### 引入建议

```python
# app/agent/graph/builder.py (重构)

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# Agent 类型定义
AgentType = Literal["coordinator", "planner", "researcher", "coder", "reporter"]

# Agent-LLM 映射
AGENT_LLM_MAP: dict[AgentType, LLMType] = {
    "coordinator": "basic",
    "planner": "reasoning",
    "researcher": "basic",
    "coder": "code",
    "reporter": "basic",
}

class AgentBuilder:
    """Agent 构建器"""

    def __init__(self):
        self.graph = StateGraph(AgentState)
        self._nodes: dict[str, callable] = {}

    def add_agent_node(
        self,
        name: str,
        agent_type: AgentType,
        tools: list,
        prompt_template: str,
    ) -> "AgentBuilder":
        """添加 Agent 节点"""
        # 创建 Agent
        agent = create_react_agent(
            name=name,
            model=LLMService.get_llm(AGENT_LLM_MAP[agent_type]),
            tools=tools,
            prompt=lambda state: render_prompt(
                prompt_template, state, state.get("locale", "zh-CN")
            ),
        )

        self._nodes[name] = agent
        return self

    def add_custom_node(self, name: str, func: callable) -> "AgentBuilder":
        """添加自定义节点"""
        self._nodes[name] = func
        return self

    def build(self):
        """构建图"""
        for name, node in self._nodes.items():
            self.graph.add_node(name, node)
        return self.graph.compile()

# 使用示例
builder = AgentBuilder()
builder.add_agent_node("coordinator", "coordinator", [], "coordinator")
builder.add_agent_node("planner", "planner", [], "planner")
builder.add_agent_node("researcher", "researcher", search_tools, "researcher")
graph = builder.build()
```

---

### 2.7 工具装饰器系统 ⭐⭐⭐

**文件**: `src/tools/decorators.py`

#### IO 日志装饰器

```python
import functools
import logging

def log_io(func: Callable) -> Callable:
    """自动记录工具输入输出"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        params = ", ".join([
            *(str(arg) for arg in args),
            *(f"{k}={v}" for k, v in kwargs.items())
        ])

        logging.info(f"Tool {func_name} called with: {params}")

        result = func(*args, **kwargs)

        logging.info(f"Tool {func_name} returned: {result}")

        return result

    return wrapper
```

#### 日志混入类

```python
class LoggedToolMixin:
    """工具日志混入类"""

    def _log_operation(self, method_name: str, *args, **kwargs):
        tool_name = self.__class__.__name__.replace("Logged", "")
        params = ", ".join([
            *(str(arg) for arg in args),
            *(f"{k}={v}" for k, v in kwargs.items())
        ])
        logging.debug(f"Tool {tool_name}.{method_name}: {params}")

    def _run(self, *args, **kwargs):
        self._log_operation("_run", *args, **kwargs)
        result = super()._run(*args, **kwargs)
        logging.debug(f"Tool returned: {result}")
        return result
```

#### 工厂函数

```python
def create_logged_tool(base_tool_class: Type[T]) -> Type[T]:
    """创建带日志的工具类"""

    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    return LoggedTool
```

#### 使用示例

```python
from src.tools.decorators import create_logged_tool
from langchain_community.tools import DuckDuckGoSearchResults

# 创建带日志的搜索工具
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)

search_tool = LoggedDuckDuckGoSearch(
    name="web_search",
    num_results=5,
)

# 自动记录输入输出
# Tool web_search called with: query="Python async", num_results=5
# Tool web_search returned: [...]
```

#### 引入建议

```python
# app/agent/tools/decorators.py (新建)

import functools
import logging
from typing import Callable, TypeVar, Type

from app.infra.log import get_logger

T = TypeVar("T")
logger = get_logger(__name__)

def log_tool_io(func: Callable) -> Callable:
    """工具输入输出日志装饰器"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__

        # 记录输入
        logger.debug(
            "tool_input",
            tool=tool_name,
            args=args,
            kwargs=kwargs,
        )

        # 执行
        try:
            result = func(*args, **kwargs)

            # 记录输出
            logger.debug(
                "tool_output",
                tool=tool_name,
                result_type=type(result).__name__,
                success=True,
            )

            return result
        except Exception as e:
            logger.error(
                "tool_error",
                tool=tool_name,
                error=str(e),
                exc_info=True,
            )
            raise

    return wrapper

# 使用示例
from app.agent.tools.builtin.search import web_search

# 装饰工具
web_search = log_tool_io(web_search)
```

---

### 2.8 RAG 集成架构 ⭐⭐⭐

**文件**: `src/rag/`

#### 支持的 RAG 后端

| 后端 | 文件 | 特性 |
|------|------|------|
| Qdrant | `qdrant.py` | 云服务 + 本地部署 |
| Milvus | `milvus.py` | 开源向量数据库 |
| RAGFlow | `ragflow.py` | 开源 RAG 引擎 |
| VikingDB | `vikingdb_knowledge_base.py` | 火山引擎 |
| Dify | `dify.py` | 开源 LLM 平台 |

#### 统一接口

```python
# src/rag/retriever.py
@dataclass
class Resource:
    """检索结果"""
    title: str
    url: str
    content: str

class Retriever:
    """统一的检索器接口"""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Resource]:
        """检索相关文档"""
        raise NotImplementedError
```

#### 配置示例

```yaml
# conf.yaml
RAG:
  provider: ragflow  # qdrant, milvus, ragflow, vikingdb, dify
  retrieval_size: 10

  # RAGFlow 配置
  ragflow:
    api_url: http://localhost:9388
    api_key: ragflow-xxx
    cross_languages: English,Chinese,Spanish

  # Qdrant 配置
  qdrant:
    location: https://xyz-example.qdrant.io:6333
    api_key: your_qdrant_api_key
    collection: documents
    embedding_provider: openai
    embedding_model: text-embedding-ada-002
```

#### 引入建议

```python
# app/agent/rag/retriever.py (新建，未来扩展)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class RetrievedDocument:
    """检索到的文档"""
    title: str
    content: str
    source: str
    score: float = 0.0

class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """检索文档"""
        pass

class RAGFlowRetriever(BaseRetriever):
    """RAGFlow 检索器"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        # 实现 RAGFlow API 调用
        pass

# 配置
# app/config/dependencies.py
def get_rag_retriever() -> BaseRetriever:
    """获取 RAG 检索器"""
    config = load_yaml_config("conf.yaml")
    rag_config = config.get("RAG", {})

    provider = rag_config.get("provider")
    if provider == "ragflow":
        return RAGFlowRetriever(
            api_url=rag_config["ragflow"]["api_url"],
            api_key=rag_config["ragflow"]["api_key"],
        )
    # 其他后端...
```

---

## 三、引入优先级与实施计划

### 3.1 优先级矩阵

| 模块 | 优先级 | 价值 | 复杂度 | 预估工时 |
|------|--------|------|--------|----------|
| Checkpoint 持久化 | P0 | ⭐⭐⭐⭐⭐ | 中 | 2-3 天 |
| LLM 服务抽象 | P0 | ⭐⭐⭐⭐ | 低 | 1 天 |
| Prompt 模板系统 | P1 | ⭐⭐⭐⭐ | 低 | 1 天 |
| 配置管理系统 | P1 | ⭐⭐⭐ | 低 | 0.5 天 |
| 工具拦截器 | P1 | ⭐⭐⭐⭐ | 中 | 1-2 天 |
| 多 Agent 架构 | P2 | ⭐⭐⭐⭐ | 高 | 3-5 天 |
| 工具装饰器 | P2 | ⭐⭐ | 低 | 0.5 天 |
| RAG 集成 | P3 | ⭐⭐⭐ | 高 | 5-7 天 |

### 3.2 分阶段实施

#### Phase 1: 核心基础设施 (1-2 天)

```
目标: 建立统一的配置和服务抽象

任务:
1. 引入配置管理系统
   - [ ] 创建 app/config/loader.py
   - [ ] 实现 YAML + 环境变量加载
   - [ ] 添加配置缓存

2. 引入 LLM 服务抽象
   - [ ] 重构 app/llm/service.py
   - [ ] 实现分层 LLM 配置
   - [ ] 添加 Token limit 推断

3. 引入 Prompt 模板系统
   - [ ] 创建 app/agent/prompts/template.py
   - [ ] 实现 Jinja2 模板渲染
   - [ ] 添加多语言支持
```

#### Phase 2: 持久化增强 (2-3 天)

```
目标: 实现生产级 Checkpoint 持久化

任务:
1. 引入 ChatStreamManager
   - [ ] 创建 app/agent/graph/checkpoint.py
   - [ ] 实现 InMemoryStore 缓存
   - [ ] 实现 PostgreSQL 持久化

2. 数据库迁移
   - [ ] 创建 chat_streams 表
   - [ ] 添加索引

3. 集成到现有流程
   - [ ] 修改 Agent 执行流程
   - [ ] 添加流式消息处理
```

#### Phase 3: 安全增强 (1-2 天)

```
目标: 实现 Human-in-the-Loop 工具控制

任务:
1. 引入工具拦截器
   - [ ] 创建 app/agent/tools/interceptor.py
   - [ ] 实现工具包装逻辑
   - [ ] 添加审批关键词识别

2. 引入工具装饰器
   - [ ] 创建 app/agent/tools/decorators.py
   - [ ] 实现 log_io 装饰器
   - [ ] 添加工具日志

3. 集成到 Agent 创建流程
   - [ ] 修改 Agent 工具绑定
   - [ ] 添加拦截配置
```

#### Phase 4: 架构升级 (可选, 3-5 天)

```
目标: 参考多 Agent 架构重构

任务:
1. 引入 AgentBuilder
   - [ ] 创建统一的 Agent 构建器
   - [ ] 实现条件边路由

2. 引入 Agent-LLM 映射
   - [ ] 定义 Agent 类型
   - [ ] 配置 LLM 映射

3. 重构图构建流程
   - [ ] 使用新构建器
   - [ ] 添加更多 Agent 类型
```

---

## 四、代码对比

### 4.1 LLM 服务对比

**DeerFlow:**
```python
# 分层 LLM + 配置合并 + 缓存
def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    env_conf = _get_env_llm_conf(llm_type)
    yaml_conf = conf.get(config_key, {})
    merged_conf = {**yaml_conf, **env_conf}

    llm = ChatOpenAI(**merged_conf)
    _llm_cache[llm_type] = llm
    return llm
```

**Kiki (当前):**
```python
# app/llm/service.py
# 简单的实例创建，无分层，无缓存
def get_llm(model_name: str) -> BaseChatModel:
    return ChatOpenAI(model=model_name)
```

### 4.2 Prompt 系统对比

**DeerFlow:**
```python
# Jinja2 模板 + 多语言 + 变量注入
def apply_prompt_template(prompt_name, state, locale):
    template = env.get_template(f"{prompt_name}.{locale}.md")
    system_prompt = template.render(**state)
    return [{"role": "system", "content": system_prompt}] + state["messages"]
```

**Kiki (当前):**
```python
# app/agent/prompts/template.py
# 简单字符串格式化
def render_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)
```

### 4.3 配置加载对比

**DeerFlow:**
```python
# YAML + 环境变量 + 缓存
def load_yaml_config(file_path: str) -> Dict:
    if file_path in _config_cache:
        return _config_cache[file_path]

    config = yaml.safe_load(open(file_path))
    processed = process_dict(config)  # 递归处理 $VAR

    _config_cache[file_path] = processed
    return processed
```

**Kiki (当前):**
```python
# app/config/dependencies.py
# Pydantic Settings + 环境变量
class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
```

---

## 五、潜在风险与注意事项

### 5.1 兼容性风险

| 风险项 | 说明 | 缓解措施 |
|--------|------|----------|
| **LangGraph 版本** | DeerFlow 使用 langgraph>=0.3.5 | 锁定版本，充分测试 |
| **psycopg 版本** | DeerFlow 使用 psycopg 3.x | 检查与 asyncpg 兼容性 |
| **Python 版本** | DeerFlow 要求 Python 3.12+ | Kiki 当前 3.11+，需升级 |

### 5.2 架构差异

| 差异点 | DeerFlow | Kiki | 影响 |
|--------|----------|------|------|
| **数据库驱动** | psycopg (同步) | asyncpg (异步) | 需要适配异步模式 |
| **Checkpoint** | 自研实现 | LangGraph 内置 | 可选择性引入 |
| **Agent 模式** | ReAct Agent | 自建图 | 架构升级成本高 |

### 5.3 实施建议

1. **渐进式引入**: 不要一次性引入所有模块
2. **充分测试**: 每个模块引入后都要有测试覆盖
3. **保持兼容**: 确保现有 API 不被破坏
4. **文档同步**: 及时更新文档和示例

---

## 六、总结

### 6.1 核心价值

DeerFlow 为 Kiki 提供了以下核心价值：

1. **生产级 Checkpoint**: 解决长期对话持久化问题
2. **统一的服务抽象**: LLM、配置、Prompt 的统一管理
3. **安全增强**: Human-in-the-Loop 工具控制
4. **可观测性**: 完善的日志和监控

### 6.2 引入建议

**推荐引入 (P0-P1)**:
- ✅ Checkpoint 持久化系统
- ✅ LLM 服务抽象层
- ✅ Prompt 模板系统
- ✅ 配置管理系统
- ✅ 工具拦截器

**可选引入 (P2-P3)**:
- ⚠️ 多 Agent 架构模式 (需要较大重构)
- ⚠️ RAG 集成 (看业务需求)
- ⚠️ 工具装饰器 (锦上添花)

### 6.3 后续行动

1. **评估讨论**: 与团队讨论引入优先级
2. **技术验证**: 对关键模块进行 PoC 验证
3. **制定计划**: 制定详细的实施计划
4. **分步实施**: 按阶段逐步引入
