# Kiki API 接口测试方案

> 对标 WeKnora99 项目
> 版本: v1.0
> 创建日期: 2025-01-30

---

## 一、测试策略概述

### 1.1 测试分层

```
┌─────────────────────────────────────────────────────┐
│  E2E 测试 (tests/e2e/)                              │
│  - 完整用户旅程：注册→登录→聊天→历史                 │
│  - Agent 工作流：创建→对话→历史→删除                 │
│  - 使用真实 PostgreSQL 数据库                        │
└─────────────────────────────────────────────────────┘
                        ↑
┌─────────────────────────────────────────────────────┐
│  集成测试 (tests/integration/)                      │
│  - API 端点测试                                      │
│  - 数据库操作测试                                    │
│  - 外部服务 Mock                                     │
└─────────────────────────────────────────────────────┘
                        ↑
┌─────────────────────────────────────────────────────┐
│  单元测试 (tests/unit/)                             │
│  - 函数级测试                                        │
│  - 业务逻辑测试                                      │
│  - 完全隔离，无外部依赖                              │
└─────────────────────────────────────────────────────┘
```

### 1.2 测试覆盖率目标

| 测试类型 | 目标覆盖率 | 当前状态 |
|----------|-----------|----------|
| 单元测试 | 80%+ | ✅ 已实现 |
| 集成测试 | API 100% | 🔄 进行中 |
| E2E 测试 | 关键流程 100% | 🔄 进行中 |

---

## 二、测试目录结构

```
tests/
├── conftest.py                      # 全局 fixtures（Mock LLM 等）
├── e2e/                             # E2E 测试（真实数据库）
│   ├── conftest.py                  # E2E fixtures
│   ├── test_full_user_journey.py    # 完整用户旅程
│   └── test_agent_workflow.py       # Agent 工作流
├── integration/                     # 集成测试
│   ├── test_api.py                  # 基础 API
│   ├── test_agents_api.py           # Agent API
│   ├── test_tools_api.py            # 工具 API
│   └── test_multi_agent_e2e.py      # 多 Agent E2E
└── unit/                            # 单元测试
    ├── test_auth_api.py
    ├── test_llm.py
    └── ...

scripts/
└── init_test_data.py                # 测试数据初始化脚本
```

---

## 三、测试场景清单

### 3.1 认证模块 (API: `/api/v1/auth/*`)

| 场景 | 测试点 | 状态 |
|------|--------|------|
| 用户注册 | 正常注册、重复邮箱、无效邮箱、弱密码 | ✅ |
| 用户登录 | 表单登录、JSON 登录、错误密码、不存在用户 | ✅ |
| Token 验证 | 有效 token、无效 token、过期 token | ✅ |
| 获取用户信息 | 带 token、不带 token、伪造 token | ✅ |
| 会话管理 | 创建、列表、删除、更新 | ✅ |
| 权限隔离 | 用户只能访问自己的会话 | ✅ |

### 3.2 聊天模块 (API: `/api/v1/chat/*`)

| 场景 | 测试点 | 状态 |
|------|--------|------|
| 同步聊天 | 正常发送、参数验证、LLM 错误处理 | ✅ |
| 流式聊天 | SSE 格式、Content-Type、事件流 | ✅ |
| 聊天历史 | 获取历史、空历史、跨会话隔离 | ✅ |
| 上下文管理 | 统计信息、清除上下文 | ✅ |

### 3.3 Agent 模块 (API: `/api/v1/agents/*`)

| 场景 | 测试点 | 状态 |
|------|--------|------|
| Swarm Agent | 创建、对话、历史、删除 | ✅ |
| Router Agent | 创建、路由对话、错误处理 | ✅ |
| Supervisor Agent | 创建、监督对话、多 Worker | ✅ |
| 工具集成 | 工具列表、工具详情、带工具 Agent | ✅ |
| 系统管理 | 列出系统、删除系统、错误处理 | ✅ |

### 3.4 工具模块 (API: `/api/v1/tools/*`)

| 场景 | 测试点 | 状态 |
|------|--------|------|
| 工具列表 | 获取所有工具、工具结构验证 | ✅ |
| 工具详情 | 获取单个工具、不存在工具 | ✅ |

---

## 四、运行测试

### 4.1 准备测试环境

```bash
# 1. 启动依赖服务
make dev-deps  # 或 docker-compose up -d postgres redis

# 2. 创建测试数据库
psql -h localhost -p 15432 -U postgres -c "CREATE DATABASE kiki_test;"

# 3. 初始化测试数据（可选）
uv run python scripts/init_test_data.py init --env testing

# 4. 设置环境变量
export KIKI_ENV=testing
export KIKI_DATABASE_URL="postgresql+asyncpg://postgres:postgres123!@#@localhost:15432/kiki_test"
```

### 4.2 运行测试命令

```bash
# 运行所有测试
uv run pytest

# 只运行 E2E 测试
uv run pytest tests/e2e/ -v

# 运行特定测试文件
uv run pytest tests/e2e/test_full_user_journey.py -v

# 运行带标记的测试
uv run pytest -m e2e -v

# 显示详细输出
uv run pytest -vv --tb=short

# 生成覆盖率报告
uv run pytest --cov=app --cov-report=html

# 跳过需要 LLM 的测试
uv run pytest -m "not llm"
```

### 4.3 按场景运行测试

```bash
# 认证流程测试
uv run pytest tests/e2e/test_full_user_journey.py::TestFullUserJourney::test_new_user_complete_flow -v

# Agent 工作流测试
uv run pytest tests/e2e/test_agent_workflow.py -v

# 流式聊天测试
uv run pytest tests/e2e/test_full_user_journey.py::TestStreamingChat -v
```

---

## 五、测试数据管理

### 5.1 初始化测试数据

```bash
# 初始化测试数据
uv run python scripts/init_test_data.py init --env testing

# 重置测试数据
uv run python scripts/init_test_data.py reset --env testing

# 清除测试数据
uv run python scripts/init_test_data.py clear --env testing

# 查看测试数据状态
uv run python scripts/init_test_data.py status --env testing
```

### 5.2 预置测试数据

| 类型 | 数量 | 说明 |
|------|------|------|
| 测试用户 | 3 | 2 个普通用户 + 1 个管理员 |
| 测试会话 | 3 | 每个用户 1-2 个会话 |
| 测试消息 | 5 | 分布在不同会话中 |

---

## 六、Mock 策略

### 6.1 LLM Mock

```python
# conftest.py 提供的 Mock LLM fixture
@pytest.fixture
def mock_llm_service():
    """返回一个 Mock LLM 服务，避免真实 API 调用"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="这是一个测试响应")
    )
    return mock_llm
```

### 6.2 数据库 Mock

```python
# 使用内存 SQLite 进行单元测试
@pytest.fixture
async def in_memory_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
```

### 6.3 Redis Mock

```python
# 使用 Mock 对象模拟 Redis
@pytest.fixture
def mock_redis():
    mock_client = MagicMock()
    mock_client.get = MagicMock(return_value=None)
    mock_client.set = MagicMock(return_value=True)
    return mock_client
```

---

## 七、CI/CD 集成

### 7.1 GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: kiki_test
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres123!@#
        ports:
          - 15432:5432
      redis:
        image: redis:7-alpine
        ports:
          - 16379:6379

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        run: uv run pytest -v --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## 八、对标 WeKnora99

### 8.1 WeKnora99 测试策略

| 特性 | WeKnora99 | Kiki |
|------|-----------|------|
| 语言 | Go | Python |
| 测试框架 | testify + httptest | pytest + TestClient |
| 数据库 | PostgreSQL + Neo4j | PostgreSQL |
| 单元测试 | ✅ 与源码同目录 | ✅ tests/unit/ |
| 集成测试 | ❌ 缺失 | ✅ tests/integration/ |
| E2E 测试 | ❌ 缺失 | ✅ tests/e2e/ |

### 8.2 Kiki 增强点

1. **完整的测试分层** - 单元、集成、E2E 三层测试
2. **真实数据库测试** - E2E 使用真实 PostgreSQL
3. **Mock LLM 服务** - 避免测试消耗 API 配额
4. **测试数据初始化脚本** - 便于准备测试环境
5. **CI/CD 集成** - 自动化测试流程

---

## 九、下一步

- [ ] 补充错误场景测试（限流、超时、网络错误）
- [ ] 添加性能测试（响应时间、并发）
- [ ] 实现视觉回归测试（前端）
- [ ] 添加安全测试（SQL 注入、XSS）
- [ ] 完善 CI/CD 测试报告

---

## 更新日志

| 日期 | 内容 | 操作者 |
|------|------|--------|
| 2025-01-30 | 创建 API 测试方案文档 | Claude |
