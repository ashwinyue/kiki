# Kiki 目录结构重构计划

## 目标

将 `core/infra` 分层架构重构为**功能驱动的平铺架构**，符合 LangGraph 应用最佳实践。

## 新目录结构

```
app/
├── agent/              # LangGraph Agent 核心能力
├── llm/                # LLM 抽象层
├── auth/               # 认证授权 (新增)
├── rate_limit/         # 限流 (新增)
├── config/             # 配置管理 (新增)
├── tools/              # 通用工具集 (新增)
├── evaluation/         # 评估框架 (从 core 移出)
├── api/                # HTTP API 层
├── models/             # 数据模型
├── repositories/       # 数据访问
├── services/           # 业务服务
├── schemas/            # Pydantic 模式
├── observability/      # 横切关注点
└── utils/              # 工具函数
```

## 文件映射表

### core/ → 拆分为多个模块

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `core/auth.py` | `auth/jwt.py` | JWT 认证 |
| `core/api_key.py` | `auth/api_key.py` | API Key 管理 |
| `core/tenant.py` | `auth/tenant.py` | 租户上下文 |
| `core/tenant_api_key.py` | `auth/tenant_api_key.py` | 租户 API Key 加密 |
| `core/tenant_middleware.py` | `auth/middleware.py` | 租户认证中间件 |
| `core/limiter.py` | `rate_limit/limiter.py` | 端点限流 |
| `core/token_bucket.py` | `rate_limit/token_bucket.py` | 令牌桶限流 |
| `core/config.py` | `config/settings.py` | 环境配置 |
| `core/configuration.py` | `config/runtime.py` | 运行时配置 |
| `core/dependencies.py` | `config/dependencies.py` | 依赖注入 |
| `core/errors.py` | `config/errors.py` | 错误处理 |
| `core/middleware.py` | `api/middleware.py` | 通用中间件 |
| `core/memory.py` | `agent/memory/context.py` | 上下文存储（合并） |
| `core/store.py` | `agent/memory/store.py` | LangGraph Store |
| `core/search.py` | `tools/search.py` | 搜索工具 |
| `core/evaluation/` | `evaluation/` | 评估框架（提升层级） |

### infra/ → tools/

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `infra/cache.py` | `tools/cache.py` | Redis 缓存 |
| `infra/redis.py` | `tools/redis.py` | Redis 客户端 |
| `infra/storage.py` | `tools/storage.py` | 对象存储 |

### 新增 __init__.py

每个新模块需要创建 `__init__.py` 并正确导出。

## 导入更新清单

涉及 **68 个文件**，需要更新以下导入模式：

```python
# 旧导入 → 新导入

# auth 模块
from app.core.auth import → from app.auth.jwt import
from app.core.api_key import → from app.auth.api_key import
from app.core.tenant import → from app.auth.tenant import

# rate_limit 模块
from app.core.limiter import → from app.rate_limit.limiter import
from app.core.token_bucket import → from app.rate_limit.token_bucket import

# config 模块
from app.core.config import → from app.config.settings import
from app.core.configuration import → from app.config.runtime import
from app.core.dependencies import → from app.config.dependencies import
from app.core.errors import → from app.config.errors import

# tools 模块
from app.infra.cache import → from app.tools.cache import
from app.infra.redis import → from app.tools.redis import
from app.infra.storage import → from app.tools.storage import
from app.core.search import → from app.tools.search import

# agent 模块
from app.core.memory import → from app.agent.memory.context import
from app.core.store import → from app.agent.memory.store import

# evaluation 模块
from app.core.evaluation import → from app.evaluation import
```

## 执行步骤

### Phase 1: 创建新目录结构
1. 创建 `app/auth/`
2. 创建 `app/rate_limit/`
3. 创建 `app/config/`
4. 创建 `app/tools/`
5. 提升 `app/evaluation/`

### Phase 2: 移动文件
1. 移动 auth 相关文件
2. 移动 rate_limit 相关文件
3. 移动 config 相关文件
4. 移动 tools 相关文件
5. 移动 evaluation 目录

### Phase 3: 更新导入
1. 更新所有 `from app.core.` 导入
2. 更新所有 `from app.infra.` 导入
3. 更新所有 `import app.core.` 导入
4. 更新测试文件

### Phase 4: 清理
1. 删除 `app/core/` 目录
2. 删除 `app/infra/` 目录
3. 更新文档

## 风险评估

| 风险 | 级别 | 缓解措施 |
|-----|------|---------|
| 导入遗漏 | 高 | 全面搜索+测试验证 |
| 循环依赖 | 中 | 保持依赖方向清晰 |
| 测试失败 | 中 | 同步更新测试导入 |
| 文档过时 | 低 | 最后统一更新 |

## 验证清单

- [ ] 所有文件移动完成
- [ ] 所有导入更新完成
- [ ] `uv run pytest` 通过
- [ ] `uv run ruff check .` 通过
- [ ] `uv run mypy app/` 通过
- [ ] 服务器正常启动

## 回滚方案

如果重构失败，使用 git 恢复：
```bash
git checkout HEAD -- app/
```
