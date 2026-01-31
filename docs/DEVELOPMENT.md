# Kiki Agent Framework - 开发指南

> 版本: v0.1.0
> 更新日期: 2025-01-30

## 目录

- [开发环境设置](#开发环境设置)
- [项目结构](#项目结构)
- [代码规范](#代码规范)
- [测试指南](#测试指南)
- [提交规范](#提交规范)
- [调试技巧](#调试技巧)
- [常见问题](#常见问题)

---

## 开发环境设置

### 环境要求

- Python 3.13+
- PostgreSQL 14+
- Redis 6+
- Docker (可选)
- Make (可选)

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/kiki.git
cd kiki

# 2. 安装依赖
uv sync

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 4. 启动开发服务器
uv run uvicorn app.main:app --reload

# 5. 访问 API 文档
open http://localhost:8000/docs
```

### Docker 开发环境

```bash
# 启动所有服务
make up

# 查看日志
make logs-app

# 进入容器
make shell

# 运行测试
make test
```

---

## 项目结构

### 目录组织

```
kiki/
├── app/                      # 应用代码
│   ├── api/                  # API 路由
│   ├── agent/                # Agent 系统
│   ├── core/                 # 核心模块
│   ├── llm/                  # LLM 服务
│   ├── infra/                # 基础设施
│   ├── observability/        # 可观测性
│   ├── models/               # 数据模型
│   ├── schemas/              # Pydantic 模式
│   ├── repositories/         # 数据访问
│   ├── services/             # 业务服务
│   ├── utils/                # 工具函数
│   └── main.py               # 应用入口
│
├── tests/                    # 测试代码
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   ├── e2e/                  # E2E 测试
│   └── conftest.py           # 测试配置
│
├── docs/                     # 文档
├── config/                   # 配置文件
├── scripts/                  # 脚本
└── pyproject.toml           # 项目配置
```

### 模块职责

| 模块 | 职责 | 规则 |
|------|------|------|
| `app/api/` | HTTP 路由处理 | 仅处理请求/响应，不含业务逻辑 |
| `app/agent/` | Agent 系统实现 | LangGraph 相关代码 |
| `app/core/` | 横切关注点 | 配置、中间件、错误处理 |
| `app/llm/` | LLM 服务抽象 | 多提供商支持 |
| `app/models/` | 数据模型 | SQLModel/SQLAlchemy |
| `app/schemas/` | 请求/响应模式 | Pydantic 模型 |
| `app/repositories/` | 数据访问 | Repository 模式 |
| `app/services/` | 业务逻辑 | 复杂业务流程 |

---

## 代码规范

### Python 代码风格

使用 **ruff** 进行代码检查：

```bash
# 检查代码
uv run ruff check .

# 自动修复
uv run ruff check --fix .
```

### 类型检查

使用 **mypy** 进行类型检查：

```bash
uv run mypy app/
```

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 模块 | 小写下划线 | `user_service.py` |
| 类 | 大驼峰 | `UserService` |
| 函数/方法 | 小写下划线 | `get_user()` |
| 常量 | 大写下划线 | `MAX_RETRIES` |
| 私有成员 | 下划线前缀 | `_internal_method` |

### 导入顺序

```python
# 1. 标准库
import os
from pathlib import Path

# 2. 第三方库
from fastapi import FastAPI
from pydantic import BaseModel

# 3. 本地模块
from app.core.config import get_settings
from app.models.user import User
```

### 文档字符串

使用 Google 风格：

```python
async def get_user(user_id: str) -> User | None:
    """根据 ID 获取用户

    Args:
        user_id: 用户 ID

    Returns:
        用户对象或 None

    Raises:
        UserNotFoundError: 用户不存在
    """
    ...
```

---

## 测试指南

### 测试结构

```
tests/
├── conftest.py              # pytest 配置和 fixtures
├── unit/                    # 单元测试
│   ├── test_config.py
│   ├── test_llm_service.py
│   └── test_agent.py
├── integration/             # 集成测试
│   ├── test_api.py
│   └── test_agents_api.py
└── e2e/                     # E2E 测试
    └── test_chat_flow.py
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/unit/test_config.py

# 运行带标记的测试
uv run pytest -m "not slow"

# 生成覆盖率报告
uv run pytest --cov=app --cov-report=html
```

### 测试标记

```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    ...

@pytest.mark.redis
@pytest.mark.postgres
async def test_database_query():
    ...

@pytest.mark.llm
async def test_llm_call():
    ...
```

### Fixtures

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    """异步 HTTP 客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def test_user():
    """测试用户数据"""
    return {
        "email": "test@example.com",
        "password": "testpass123",
    }
```

### 测试示例

```python
import pytest
from app.core.config import get_settings

class TestConfig:
    """配置测试"""

    def test_get_settings_singleton(self):
        """测试配置单例"""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_environment_detection(self, monkeypatch):
        """测试环境检测"""
        monkeypatch.setenv("KIKI_ENV", "production")
        from app.core.config import reload_settings
        settings = reload_settings()
        assert settings.environment.value == "production"

@pytest.mark.asyncio
class TestLLMService:
    """LLM 服务测试"""

    async def test_call_with_retry(self, llm_service):
        """测试重试机制"""
        # 使用 mock LLM
        ...
```

---

## 提交规范

### Commit Message 格式

```
<type>: <description>

[optional body]
```

### 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `refactor` | 重构代码 |
| `docs` | 文档变更 |
| `test` | 测试相关 |
| `chore` | 构建/工具变更 |
| `perf` | 性能优化 |
| `ci` | CI 配置 |

### 示例

```bash
# 简单提交
git commit -m "feat: 添加用户注册接口"

# 带详细说明
git commit -m "fix: 修复检查点持久化问题

- 修复 PostgreSQL 连接池未正确关闭的问题
- 添加连接池健康检查
- 优化检查点保存性能"
```

---

## 调试技巧

### 1. 日志调试

```python
from app.observability.logging import get_logger

logger = get_logger(__name__)

logger.debug(
    "debug_message",
    variable=value,
    context={"extra": "info"},
)
```

### 2. 可视化图结构

```python
from app.agent.graphs import ChatGraph

graph = ChatGraph()
compiled = graph.compile()

# 打印图结构
print(compiled.get_graph().print_ascii())

# 生成 Mermaid 图表
mermaid = compiled.get_graph().draw_mermaid()
print(mermaid)
```

### 3. 追踪状态变化

```python
async for chunk in graph.astream(
    input_data,
    config,
    stream_mode="values",
):
    print("状态:", chunk)
```

### 4. LangSmith 追踪

```bash
# 设置环境变量
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key

# 查看追踪
# https://smith.langchain.com
```

---

## 常见问题

### Q: 如何添加新的 LLM 提供商？

1. 在 `app/llm/registry.py` 中注册新模型
2. 在 `app/llm/providers.py` 中实现提供商逻辑
3. 更新配置文件添加新选项

### Q: 如何创建自定义工具？

使用 `@tool` 装饰器：

```python
from app.agent.tools import tool

@tool
async def my_tool(param: str) -> str:
    """工具描述"""
    return f"结果: {param}"
```

### Q: 如何修改数据库模型？

1. 修改 `app/models/` 中的模型
2. 创建迁移脚本
3. 运行迁移

### Q: 如何添加新的 API 端点？

1. 在 `app/api/v1/` 中创建新文件
2. 定义路由和处理函数
3. 在 `app/api/v1/__init__.py` 中注册

### Q: 测试失败怎么办？

```bash
# 查看详细输出
uv run pytest -vvs

# 进入调试器
uv run pytest --pdb

# 只运行失败的测试
uv run pytest --lf
```

---

## 贡献流程

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交变更 (`git commit -m 'feat: xxx'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

### PR 检查清单

- [ ] 代码通过 ruff 检查
- [ ] 代码通过 mypy 检查
- [ ] 测试覆盖率 > 80%
- [ ] 添加了必要的文档
- [ ] 更新了 CHANGELOG

---

## 参考资源

- [项目架构文档](ARCHITECTURE.md)
- [Agent 系统文档](AGENT.md)
- [API 文档](API.md)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
