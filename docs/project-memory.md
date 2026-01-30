# Kiki 项目记忆

## 项目概述

**项目名称**: Kiki Agent Framework
**版本**: 0.1.0
**目标**: 基于 FastAPI + LangGraph 的企业级 Python Agent 开发脚手架

---

## 当前开发需求

### 需求标题：Python Agent 企业级开发脚手架

### 参考项目

#### 1. WeKnora99
- **路径**: `aold/WeKnora99`
- **技术栈**: Go + Gin + gRPC + PostgreSQL + Neo4j + Redis
- **定位**: 企业级 RAG 框架，微服务架构
- **核心优势**:
  - 高性能并发处理
  - 完善的企业级功能（多租户、权限管理）
  - 知识图谱支持 (Neo4j)
  - 完整的监控体系 (OpenTelemetry)
  - 事件驱动架构
  - 依赖注入容器

#### 2. fastapi-langgraph-agent-production-ready-template2
- **路径**: `aold/fastapi-langgraph-agent-production-ready-template2`
- **技术栈**: Python 3.13+ + FastAPI + LangGraph + PostgreSQL + Langfuse
- **定位**: 生产就绪的 AI Agent 开发模板
- **核心优势**:
  - LangGraph 深度集成
  - 开发效率高
  - AI 生态丰富
  - 快速原型开发
  - 配置简单

---

## 脚手架设计目标

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 异步支持、类型提示、AI 生态 |
| Agent 框架 | LangGraph | 工作流编排、状态管理、检查点 |
| 数据库 | PostgreSQL + pgvector | 关系型 + 向量存储 |
| 缓存 | Redis | 高性能缓存、分布式锁 |
| 监控 | Langfuse + Prometheus | AI 追踪 + 系统指标 |
| 包管理 | uv | 快速依赖解析 |
| 部署 | Docker + Docker Compose | 容器化部署 |

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Kiki Agent Framework                 │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                        │
│  ├── Middleware (Auth, CORS, Logging, RateLimit)           │
│  └── Routes (v1 API)                                        │
├─────────────────────────────────────────────────────────────┤
│  Service Layer                                              │
│  ├── Agent Service (LangGraph orchestration)               │
│  ├── RAG Service (Retrieval Augmented Generation)          │
│  ├── Memory Service (Long-term/Short-term)                 │
│  └── Tool Service (Function calling)                       │
├─────────────────────────────────────────────────────────────┤
│  Core Layer                                                 │
│  ├── Config (Environment-based)                            │
│  ├── Logging (Structured logging)                          │
│  ├── LLM (Model registry, retry, fallback)                 │
│  └── Graph (StateGraph, Command, Checkpoint)               │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── PostgreSQL (Business data)                            │
│  ├── pgvector (Vector embeddings)                          │
│  └── Redis (Cache, session)                                │
└─────────────────────────────────────────────────────────────┘
```

### 核心特性

#### 1. 企业级功能
- [ ] 认证授权 (JWT + RBAC)
- [ ] 请求限流 (Token bucket / Redis)
- [ ] 审计日志
- [ ] 多租户支持
- [ ] 配置管理 (多环境)

#### 2. AI 原生设计
- [ ] LangGraph 工作流编排
- [ ] Agent 状态管理
- [ ] 检查点持久化
- [ ] RAG 流水线
- [ ] 记忆管理 (短期/长期)

#### 3. 可观测性
- [ ] 结构化日志 (structlog)
- [ ] 指标收集 (Prometheus)
- [ ] 分布式追踪 (Langfuse)
- [ ] 性能监控

#### 4. 开发体验
- [ ] 类型提示 (mypy)
- [ ] 代码格式化 (ruff)
- [ ] 测试框架 (pytest)
- [ ] 热重载开发
- [ ] CLI 工具

---

## 可复用设计模式

### 来自 WeKnora99
1. **事件驱动架构** - 解耦业务逻辑
2. **依赖注入容器** - 管理复杂依赖
3. **中间件模式** - 横切关注点
4. **仓储模式** - 数据访问抽象

### 来自 FastAPI LangGraph
1. **分层架构** - API/Service/Data 分离
2. **服务模式** - 业务逻辑封装
3. **工厂模式** - 配置创建
4. **Pydantic 验证** - 请求/响应验证

---

## 待实现功能清单

### Phase 1: 核心框架
- [ ] 配置管理完善
- [ ] 日志系统完善
- [ ] 中间件实现 (日志、指标、限流、CORS)
- [ ] LLM 服务 (模型注册表、重试、回退)

### Phase 2: Agent 核心
- [ ] LangGraph 集成
- [ ] 状态管理
- [ ] 检查点持久化
- [ ] 工具注册

### Phase 3: 企业功能
- [ ] 认证授权
- [ ] RAG 流水线
- [ ] 记忆管理
- [ ] 监控集成

### Phase 4: 开发工具
- [ ] CLI 命令
- [ ] 项目模板
- [ ] 测试脚手架
- [ ] 文档生成

---

## 参考代码位置

| 功能 | 参考路径 |
|------|----------|
| 事件驱动 | `aold/WeKnora99/internal/event/` |
| 依赖注入 | `aold/WeKnora99/internal/container/` |
| LangGraph | `aold/fastapi-langgraph-agent-production-ready-template2/app/core/langgraph/` |
| 监控 | `aold/WeKnora99/internal/tracing/` |

---

## 更新日志

| 日期 | 内容 | 操作者 |
|------|------|--------|
| 2025-01-30 | 创建项目记忆，记录脚手架开发需求 | Claude |
