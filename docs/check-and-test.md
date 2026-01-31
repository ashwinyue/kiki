# Kiki 全面检查与测试指南

> 适用范围：后端 FastAPI + LangGraph、前端 React、依赖服务（PostgreSQL/Redis）、LLM/工具生态  
> 最后更新：2026-01-30

---

## 1. 目标与原则

- **目标**：在没有充分测试基础上，建立一套可重复执行的“检查 + 测试”流程，尽快发现接口不可用、配置错误、依赖异常、逻辑回归等问题。
- **原则**：先健康检查与冒烟，再集成与 E2E，最后覆盖性能与安全；结果可记录、可复现、可对比。

---

## 2. 快速健康检查（5–10 分钟）

用于判断“服务是否基本可用”，适合每日/每次部署后执行。

### 2.1 启动依赖服务

```bash
make dev-deps
```

### 2.2 启动后端

```bash
make dev-backend
```

### 2.3 基础健康端点

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics
curl -s http://localhost:8000/openapi.json | head
```

**期望结果**
- `/health` 返回 `status=healthy` 或 `degraded` 且包含 `redis` 状态。
- `/metrics` 返回 Prometheus 文本。
- `/openapi.json` 返回 OpenAPI 文档。

---

## 3. 环境与依赖检查（一次性/变更时）

### 3.1 配置文件与环境变量

- 参考 `.env.example` 与 `.env.testing`。
- 测试环境推荐使用 `.env.testing`（LLM 为 mock，避免真实 API）。
- 必填项（常见）：
  - `KIKI_DATABASE_URL`
  - `KIKI_REDIS_URL`
  - `KIKI_LLM_PROVIDER` / `KIKI_LLM_MODEL`
  - `KIKI_JWT_SECRET`

### 3.2 依赖服务可用性

```bash
docker ps
```

确保 PostgreSQL 与 Redis 运行中，端口与配置匹配。

### 3.3 数据库健康

如需迁移，Docker 启动参数 `RUN_MIGRATIONS=true` 会尝试执行 `alembic upgrade head`。

建议执行一次数据库连通性验证（示例）：

```bash
psql "postgresql://postgres:postgres@localhost:5432/kiki_test" -c "select 1;"
```

---

## 4. 代码质量检查（变更后）

```bash
make backend-lint
```

**包含：**
- ruff 检查
- ruff 格式化检查
- mypy 类型检查

---

## 5. 自动化测试（建议执行顺序）

### 5.1 单元测试（最先）

```bash
uv run pytest tests/unit -v
```

### 5.2 集成测试

```bash
uv run pytest tests/integration -v
```

### 5.3 E2E 测试（后端）

```bash
uv run pytest tests/e2e -v
```

### 5.4 全量测试

```bash
make backend-test
```

> 详见 `docs/api-testing.md` 与 `docs/e2e-testing.md`

---

## 6. 接口连通性检查（手工/自动）

覆盖 **业务关键路径**，优先验证“可达 + 认证 + 失败可控”。

### 6.1 关键 API 模块

- 认证：`/api/v1/auth/*`
- 聊天：`/api/v1/chat/*`
- Agent：`/api/v1/agents/*`
- 工具：`/api/v1/tools/*`
- API Keys：`/api/v1/api_keys/*`
- MCP 服务：`/api/v1/mcp_services/*`
- 流式：`/api/v1/chat/stream`（LangGraph SSE）
- 评估：`/api/v1/evaluation/*`
- 租户：`/api/v1/tenants/*`

### 6.2 手工验证清单（示例）

- 认证
  - 注册/登录是否返回 token
  - token 是否可访问受保护接口
  - token 过期/非法时返回合理错误
- 聊天
  - 普通 chat 是否可用
  - 流式 chat 是否输出完整事件流
  - 历史/会话是否隔离
- Agent
  - Router/Supervisor/Swarm 是否可创建与对话
  - 工具调用失败是否可控（错误信息一致、无 500 泄漏）
- 工具 & MCP
  - 工具列表/详情返回结构完整
  - MCP 服务的注册/加载是否可用
- 多租户/ApiKey
  - 租户隔离是否有效
  - API Key 的创建/吊销是否生效

---

## 7. 可观测性与指标检查

### 7.1 日志

- 日志是否为结构化格式（lowercase_underscore）
- 错误路径是否使用 `log.exception(...)` 输出堆栈

### 7.2 Prometheus 指标

```bash
curl -s http://localhost:8000/metrics | head
```

**检查点**
- 是否有 LLM 请求计数与耗时指标
- 是否有错误率指标

---

## 8. 性能与稳定性（可选增强）

### 8.1 基础并发压测（可选）

使用 `hey`/`wrk` 等工具对关键接口进行 1–5 分钟压测，记录：
- P50/P95 响应时间
- 错误率
- 资源占用

### 8.2 限流与降级验证

- 触发限流时是否返回标准错误结构
- Redis 不可用时 `/health` 是否返回 `degraded`

---

## 9. 安全与合规检查（可选增强）

- JWT Secret 是否满足长度要求
- CORS 是否仅允许必要域名
- 敏感信息是否被日志脱敏
- SQL 注入/XSS 基础验证（工具 + 人工）

---

## 10. 前端检查（如有）

### 10.1 启动与构建

```bash
make frontend-install
make frontend-run
make frontend-build
```

### 10.2 前端质量检查

```bash
make frontend-lint
```

> 若执行前端 E2E（Playwright 规划），参考 `docs/e2e-testing.md`。

---

## 11. 回归测试清单（每次发布前）

- 核心路径：注册 → 登录 → 创建会话 → 聊天 → 历史
- Agent：Router/Supervisor/Swarm 各走一遍
- 工具调用：至少 1 个带工具请求
- 健康检查：`/health` 与 `/metrics`
- 日志与指标：是否有预期输出

---

## 12. 结果记录建议（模板）

建议每次执行后记录以下信息，便于回溯与对比：

```
日期：
版本/分支：
环境：
依赖服务状态：
健康检查：
单元/集成/E2E 结果：
失败用例与日志：
回归结论：
```

---

## 相关文档

- `docs/api-testing.md`
- `docs/e2e-testing.md`
- `ENTERPRISE_GUIDE.md`
- `DEPLOYMENT.md`
