# 企业级 Agent 开发脚手架 - Monorepo 架构实施计划

## 项目概述

基于 `fastapi-l` (LangGraph Agent 模板) 和 `Weknora` (企业级架构)，构建一个 **业务与包分离** 的企业级 Agent 开发脚手架。

### 目标
- 4 个核心可复用包（可独立发布到 PyPI）
- 2 个示例应用（minimal + production-ready）
- 面向开源社区（开箱即用）
- Monorepo 结构（使用 uv workspace）

---

## 目录结构

```
kiki/
├── pyproject.toml              # Workspace 根配置
├── uv.lock                     # 统一锁文件
├── README.md
├── AGENTS.md                   # 已有的架构指南
│
├── packages/                   # 核心可复用包
│   ├── kiki-core/              # 包1: LangGraph 集成层
│   │   ├── src/kiki_core/
│   │   │   ├── graph.py        # GraphBuilder、CompiledGraph
│   │   │   ├── state.py        # AgentState、StateSchema
│   │   │   ├── checkpoint.py   # CheckpointManager
│   │   │   ├── tools.py        # ToolRegistry、@tool 装饰器
│   │   │   └── agent.py        # BaseAgent 抽象
│   │   └── pyproject.toml
│   │
│   ├── kiki-config/            # 包2: 配置管理系统
│   │   ├── src/kiki_config/
│   │   │   ├── settings.py     # BaseSettings、get_settings()
│   │   │   ├── env.py          # 环境检测、load_dotenv
│   │   │   └── schema.py       # DatabaseConfig、LLMConfig 等
│   │   └── pyproject.toml
│   │
│   ├── kiki-auth/              # 包3: 认证授权模块
│   │   ├── src/kiki_auth/
│   │   │   ├── jwt.py          # JWTManager
│   │   │   ├── session.py      # SessionManager
│   │   │   ├── api_key.py      # APIKeyManager
│   │   │   ├── permissions.py  # RBAC
│   │   │   └── dependencies.py # FastAPI Depends
│   │   └── pyproject.toml
│   │
│   └── kiki-observability/     # 包4: 可观测性栈
│       ├── src/kiki_observability/
│       │   ├── logging.py      # structlog 配置
│       │   ├── langfuse.py     # Langfuse 集成
│       │   ├── metrics.py      # Prometheus 指标
│       │   └── middleware.py   # ObservabilityMiddleware
│       └── pyproject.toml
│
├── apps/                       # 示例应用
│   ├── minimal/                # 最小示例（50 行代码）
│   │   ├── src/minimal/
│   │   │   └── main.py
│   │   └── pyproject.toml
│   │
│   └── production-ready/       # 生产级示例
│       ├── src/app/
│       │   ├── main.py
│       │   ├── api/v1/         # 路由
│       │   ├── graph/          # Agent 定义
│       │   └── models/         # 数据模型
│       ├── docker/
│       │   ├── Dockerfile
│       │   └── docker-compose.yml
│       └── pyproject.toml
│
├── scripts/                    # 工具脚本
│   ├── release.sh              # 发布脚本
│   └── sync_versions.py        # 版本同步
│
└── tests/                      # 集成测试
```

---

## 包依赖关系

```
┌─────────────────────────────────────────┐
│           apps/* (应用层)                │
├─────────────────────────────────────────┤
│           kiki-core                      │
│     (LangGraph 工作流集成)               │
├─────────────────────────────────────────┤
│  kiki-config │ kiki-auth │ kiki-obs     │
│   (配置)      (认证)      (可观测)       │
└─────────────────────────────────────────┘
```

**依赖层级**:
- L0: `kiki-config`, `kiki-auth`, `kiki-observability` (无内部依赖)
- L1: `kiki-core` (依赖 `kiki-config`)
- L2: `apps/*` (依赖所有包)

---

## 实施阶段

### 阶段 1: 基础设施搭建
1. 创建根目录 `pyproject.toml` (uv workspace 配置)
2. 创建 `packages/` 和 `apps/` 目录结构
3. 配置统一工具链 (ruff, pytest, coverage)

### 阶段 2: 核心包开发 (按依赖顺序)
1. **kiki-config** - 配置管理 (最快，无依赖)
2. **kiki-observability** - 可观测性 (无依赖)
3. **kiki-auth** - 认证授权 (无依赖)
4. **kiki-core** - LangGraph 集成 (依赖 kiki-config)

### 阶段 3: 示例应用
1. **minimal** - 50 行代码演示
2. **production-ready** - 完整功能示例

### 阶段 4: 测试与文档
1. 为每个包编写单元测试
2. 集成测试覆盖
3. README 和 API 文档

### 阶段 5: 发布准备
1. 配置 CI/CD
2. 发布脚本
3. PyPI 发布

---

## 关键文件清单

| 文件 | 作用 | 优先级 |
|-----|------|-------|
| `/pyproject.toml` | Workspace 根配置 | P0 |
| `/packages/kiki-config/src/kiki_config/settings.py` | 配置管理核心 | P0 |
| `/packages/kiki-observability/src/kiki_observability/middleware.py` | 统一可观测中间件 | P0 |
| `/packages/kiki-auth/src/kiki_auth/dependencies.py` | FastAPI 认证依赖注入 | P0 |
| `/packages/kiki-core/src/kiki_core/graph.py` | GraphBuilder 抽象 | P0 |
| `/apps/minimal/src/minimal/main.py` | 最小示例 | P1 |
| `/apps/production-ready/src/app/main.py` | 生产示例 | P1 |

---

## 验证方式

### 功能验证
```bash
# 1. 安装依赖
uv sync

# 2. 运行最小示例
cd apps/minimal
uv run python -m minimal.main

# 3. 运行测试
uv run pytest

# 4. 构建包
for pkg in packages/*/; do uv build "$pkg"; done
```

### 集成验证
- [ ] 最小示例可以独立运行
- [ ] 生产示例包含完整 CRUD
- [ ] 所有包可以独立发布到 PyPI
- [ ] Docker 镜像可以构建成功
