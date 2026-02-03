# YAML 配置系统实现完成

**日期**: 2026-02-04
**参考**: DeerFlow 配置管理设计

## 实现概览

参考 DeerFlow 的配置管理设计，实现了完整的 YAML + 环境变量配置系统。

### 核心特性

1. **多源配置加载**（优先级从高到低）：
   - 环境变量（`KIKI_*`）
   - YAML 配置文件（`conf.yaml`）
   - Pydantic 默认值

2. **环境变量替换语法**：
   - `$VAR_NAME` - 简单替换
   - `${VAR_NAME:-default}` - 带默认值替换

3. **分层 LLM 配置**：
   - `REASONING_MODEL` - 推理模型（DeepSeek Reasoner）
   - `BASIC_MODEL` - 基础模型（GPT-4o）
   - `CODE_MODEL` - 代码模型（Claude Sonnet）
   - `VISION_MODEL` - 视觉模型（GPT-4o）

4. **Agent-LLM 映射**：
   - 支持从 YAML 自定义 Agent 与 LLM 类型的映射
   - 默认映射：planner→reasoning, coder→code, researcher→basic 等

## 文件变更

### 新增文件

#### `app/config/loader.py`
YAML + 环境变量配置加载器

```python
@lru_cache(maxsize=32)
def load_yaml_config(file_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件（带缓存）"""

def _process_env_vars(config: Any) -> Any:
    """递归处理配置中的环境变量"""

# 类型安全的辅助函数
def get_str_env(name: str, default: str = "") -> str
def get_int_env(name: str, default: int = 0) -> int
def get_bool_env(name: str, default: bool = False) -> bool
def get_float_env(name: str, default: float = 0.0) -> float
def get_list_env(name: str, default: list[str] | None = None, separator: str = ",") -> list[str]
```

#### `conf.example.yaml`
完整的配置示例文件

包含：
- 应用配置（APP_NAME, DEBUG, ENVIRONMENT）
- 服务器配置（HOST, PORT, WORKERS）
- 数据库配置（DATABASE_URL, pool settings）
- 分层 LLM 配置（BASIC_MODEL, REASONING_MODEL, CODE_MODEL, VISION_MODEL）
- 搜索引擎配置（Tavily, DuckDuckGo）
- 工具配置（Python REPL, Web Crawler）
- Agent LLM 映射
- Multi-Agent 配置
- Checkpoint 持久化配置
- CORS、安全、可观测性配置

### 修改文件

#### `app/config/settings.py`
添加 YAML 配置加载支持

```python
CONFIG_FILE = Path("conf.yaml")

def _load_yaml_config() -> dict[str, Any]:
    """从 YAML 文件加载配置"""

def _merge_config_with_defaults(...) -> dict[str, Any]:
    """合并 YAML 配置与默认值"""

def get_settings() -> Settings:
    """获取配置实例（支持 YAML 加载）"""
```

#### `app/agent/config/llm_config.py`
添加 YAML 配置加载支持

```python
def _load_config_from_yaml() -> None:
    """从 YAML 配置文件加载 LLM 配置"""

def _load_agent_llm_mapping() -> dict[str, LLMType]:
    """从 YAML 加载 Agent-LLM 映射"""
```

#### `app/config/__init__.py`
导出 YAML 加载器函数

```python
from app.config.loader import (
    get_bool_env,
    get_float_env,
    get_int_env,
    get_list_env,
    get_str_env,
    load_yaml_config,
    reload_config,
)
```

#### `.gitignore`
添加 `conf.yaml` 到忽略列表（包含敏感信息）

## 配置优先级

```
环境变量 > YAML 配置 > 默认值
```

示例：
```bash
# conf.yaml
DATABASE_URL: "postgresql://localhost:5432/kiki"

# 环境变量
export KIKI_DATABASE_URL="postgresql://prod-server:5432/kiki"

# 结果：使用环境变量的值
```

## 环境变量替换

YAML 配置支持环境变量替换：

```yaml
# 简单替换
api_key: "$OPENAI_API_KEY"

# 带默认值
base_url: "${API_BASE_URL:-https://api.openai.com}"
```

## 使用示例

### 1. 创建配置文件

```bash
cp conf.example.yaml conf.yaml
# 编辑 conf.yaml，填写实际配置
```

### 2. 在代码中使用

```python
from app.config.settings import get_settings
from app.agent.config import get_llm_for_agent, AGENT_LLM_MAP

# 获取配置
settings = get_settings()

# 获取 Planner 的 LLM（会根据 YAML 配置自动选择）
planner_llm = get_llm_for_agent("planner")
```

### 3. 环境变量覆盖

```bash
# 覆盖 LLM 配置
export KIKI_LLM__REASONING__MODEL="gpt-4o"
export KIKI_LLM__BASIC__PROVIDER="openai"

# 覆盖 Agent-LLM 映射（需要在 YAML 中配置）
```

## 测试验证

所有功能已通过测试：

```bash
✓ LLM 配置模块测试通过
✓ Settings 模块测试通过
✓ YAML 配置加载器测试通过
```

## DeerFlow 对比

| 功能 | DeerFlow | Kiki |
|------|----------|------|
| YAML 配置 | ✅ | ✅ |
| 环境变量替换 | ✅ | ✅ |
| 分层 LLM 配置 | ✅ | ✅ |
| Agent-LLM 映射 | ✅ | ✅ |
| 配置缓存 | ✅ | ✅ |

## 下一步

根据 DeerFlow 分析报告，剩余功能：

1. ✅ YAML 配置管理系统（已完成）
2. ✅ 分层 LLM 配置（已完成）
3. ⏳ 完善 Checkpoint 持久化
4. ⏳ 文档更新
5. ⏳ 测试集成
