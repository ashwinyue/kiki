# Phase 2 核心功能实施报告

> 实施日期: 2025-02-04
> 参考: `.reports/multi-agent-architecture-improvement.md`

## 概述

完成了多 Agent 架构改进的 Phase 2：核心功能改进。包括 Jinja2 提示词模板集成和专门化角色支持。

## 实施的文件

### 修改的文件

| 文件 | 变更 |
|------|------|
| `app/agent/prompts/template.py` | 添加专门化角色模板（planner, researcher, analyst, coder, reporter） |
| `app/agent/graph/agent_factory.py` | 集成 Jinja2 模板渲染，添加 locale 参数 |

## 核心功能

### 1. 专门化角色模板

添加了 5 个新的提示词模板，支持中英文：

#### Planner 模板

```jinja2
你是一个任务规划专家，负责将复杂目标分解为可执行的步骤。

你的职责：
1. 分析用户的目标和需求
2. 识别需要完成的主要任务
3. 将任务分解为清晰的步骤
4. 考虑任务之间的依赖关系
5. 确定最优的执行顺序

当前时间：{{ now().strftime('%Y-%m-%d %H:%M') }}

用户目标：{{ goal }}
{% if context %}
背景信息：{{ context }}
{% endif %}
```

#### Researcher 模板

```jinja2
你是一个信息检索专家，负责查找和验证相关信息。

你的职责：
1. 理解用户的信息需求
2. 使用搜索工具查找相关信息
3. 验证信息来源的可靠性
4. 整理和总结搜索结果
5. 引用信息来源

可用工具：
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

搜索目标：{{ query }}
```

#### Analyst 模板

```jinja2
你是一个数据分析专家，负责分析和解读数据。

你的职责：
1. 理解数据分析的目标
2. 应用合适的分析方法
3. 识别数据中的模式和趋势
4. 得出合理的结论
5. 提供可操作的建议

分析目标：{{ goal }}
```

#### Coder 模板

```jinja2
你是一个代码专家，负责编写和优化代码。

你的职责：
1. 理解编程需求
2. 编写高质量、可维护的代码
3. 解释代码逻辑
4. 调试和优化代码
5. 遵循编程最佳实践

可用工具：
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

编程任务：{{ task }}
```

#### Reporter 模板

```jinja2
你是一个报告生成专家，负责聚合结果并生成结构化报告。

你的职责：
1. 聚合各个 Agent 的输出
2. 生成结构化的报告
3. 突出关键发现和结论
4. 提供可操作的建议
5. 确保报告清晰易读

报告主题：{{ topic }}
{% if outputs %}
各 Agent 输出：
{% for agent_name, output in outputs.items() %}
{{ agent_name }}: {{ output }}
{% endfor %}
{% endif %}
```

### 2. Agent 工厂集成

#### 新增参数

```python
async def create_agent(
    # ... 其他参数 ...
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    locale: str = "zh-CN",  # 新增：语言区域参数
    **llm_kwargs,
)
```

#### 模板渲染逻辑

```python
if prompt_template:
    # 使用 Jinja2 模板渲染提示词
    try:
        template_vars = {
            "agent_name": agent_name,
            "tools": processed_tools,
            "locale": locale,
        }
        system_prompt_text = render_prompt(
            prompt_template,
            locale=locale,
            **template_vars,
        )
        prompt = _build_prompt(system_prompt_text, state_modifier)
    except Exception as e:
        # 回退到默认提示词
        prompt = _build_prompt(f"你是 {agent_name}。", state_modifier)
elif system_prompt:
    # 使用硬编码提示词
    prompt = _build_prompt(system_prompt, state_modifier)
else:
    # 使用默认提示词
    prompt = _build_prompt(f"你是 {agent_name}。", state_modifier)
```

#### 专门化角色函数更新

所有专门化角色函数现在都使用模板作为默认值：

```python
async def create_planner_agent(
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,  # 覆盖模板
    locale: str = "zh-CN",
    **kwargs,
) -> Any:
    return await create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=tools or [],
        prompt_template=None if system_prompt else "planner",  # 默认使用模板
        system_prompt=system_prompt,  # 覆盖模板
        locale=locale,
        **kwargs,
    )
```

## 使用示例

### 基础用法（使用默认模板）

```python
from app.agent.graph import create_planner_agent

# 使用默认模板（中文）
planner = await create_planner_agent()

# 使用英文模板
planner_en = await create_planner_agent(locale="en-US")
```

### 自定义提示词（覆盖模板）

```python
# 使用自定义提示词
planner = await create_planner_agent(
    system_prompt="你是一个敏捷开发规划专家。"
)
```

### 使用指定模板

```python
from app.agent.graph import create_agent

# 直接指定模板名称
planner = await create_agent(
    agent_name="my-planner",
    agent_type="planner",
    prompt_template="planner",  # 使用 planner 模板
    locale="en-US",  # 使用英文
)
```

### 带变量的模板

```python
# 传递变量到模板
researcher = await create_agent(
    agent_name="researcher",
    agent_type="researcher",
    prompt_template="researcher",
    tools=[tavily_search],
)
# 模板会自动填充 {{ tools }} 变量
```

### 国际化支持

```python
# 中文
coder_zh = await create_coder_agent(
    tools=[python_repl],
    locale="zh-CN",
)

# 英文
coder_en = await create_coder_agent(
    tools=[python_repl],
    locale="en-US",
)
```

## 设计决策

### 1. 模板优先级

| 优先级 | 方式 | 说明 |
|--------|------|------|
| 1 | `prompt_template` | 使用 Jinja2 模板（推荐） |
| 2 | `system_prompt` | 直接使用硬编码提示词 |
| 3 | 默认 | "你是 {agent_name}。" |

### 2. 为什么使用 Jinja2

| 优势 | 说明 |
|------|------|
| **标准化** | 行业标准模板引擎 |
| **灵活性** | 支持条件、循环等逻辑 |
| **国际化** | 易于实现多语言支持 |
| **可维护性** | 模板与代码分离 |

### 3. 模板变量

每个模板可用的变量：

| 变量 | 类型 | 说明 |
|------|------|------|
| `agent_name` | str | Agent 名称 |
| `tools` | list | 工具列表 |
| `locale` | str | 语言区域 |
| `now()` | callable | 当前时间函数 |
| `today()` | callable | 当前日期函数 |

## 后续步骤

### Phase 3: 高级功能 (2-3 天)

```
1. 工具拦截器集成
   - 增强现有工具拦截器
   - 集成到 Agent 工厂

2. LLM 智能路由
   - 意图分类
   - 集成到 Supervisor 模式
```

## 测试

### 单元测试（待添加）

```python
# tests/unit/test_agent_factory_with_templates.py
async def test_create_planner_with_template():
    planner = await create_planner_agent()
    assert planner is not None

async def test_create_with_locale():
    planner_en = await create_planner_agent(locale="en-US")
    planner_zh = await create_planner_agent(locale="zh-CN")
    assert planner_en is not None
    assert planner_zh is not None

async def test_custom_prompt_overrides_template():
    planner = await create_planner_agent(
        system_prompt="自定义提示词"
    )
    assert planner is not None
```

## 风险和注意事项

| 风险 | 缓解措施 |
|------|----------|
| **模板复杂度** | 保持模板简洁，避免过度嵌套 |
| **翻译质量** | 提供人工翻译的模板 |
| **变量错误** | 模板渲染失败时回退到默认 |

## 参考

- Jinja2 文档: https://jinja.palletsprojects.com/
- DeerFlow 提示词模板: `.reports/multi-agent-architecture-improvement.md`
