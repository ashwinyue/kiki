# Kiki 知识库与模型管理 API 实施计划

## 概述

基于现有代码结构，添加知识库管理 API 和模型管理 API，完全对齐 WeKnora99 的接口设计。

## 现有能力分析

### 已有组件（可复用）

| 组件 | 位置 | 功能 |
|------|------|------|
| `BaseRepository` | `app/repositories/base.py` | 提供租户隔离、分页、软删除支持 |
| `PgVectorStore` | `app/agent/capabilities/rag.py` | 向量存储与相似度搜索 |
| `get_embeddings` | `app/llm/embeddings.py` | 多提供商嵌入服务 |
| 数据模型 | `app/models/knowledge.py` | Model, KnowledgeBase, Knowledge, Chunk 已定义 |

### 待实现组件

| 组件 | 文件路径 | 功能 |
|------|----------|------|
| Schema | `app/schemas/knowledge.py` | 知识库请求/响应模型 |
| Schema | `app/schemas/model.py` | 模型请求/响应模型 |
| Repository | `app/repositories/knowledge.py` | 知识库数据访问 |
| Repository | `app/repositories/model.py` | 模型数据访问 |
| Service | `app/services/knowledge_service.py` | 知识库业务逻辑 |
| Service | `app/services/model_service.py` | 模型业务逻辑 |
| API | `app/api/v1/knowledge.py` | 知识库 HTTP 接口 |
| API | `app/api/v1/models.py` | 模型 HTTP 接口 |

## API 接口清单

### 知识库管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases` | 查询知识库列表 |
| GET | `/api/v1/knowledge-bases/{kb_id}` | 获取知识库详情 |
| PUT | `/api/v1/knowledge-bases/{kb_id}` | 更新知识库 |
| DELETE | `/api/v1/knowledge-bases/{kb_id}` | 删除知识库 |
| POST | `/api/v1/knowledge-bases/{kb_id}/hybrid-search` | 混合搜索 |

### 知识条目管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/knowledge-bases/{kb_id}/knowledge/file` | 从文件创建 |
| POST | `/api/v1/knowledge-bases/{kb_id}/knowledge/url` | 从 URL 创建 |
| GET | `/api/v1/knowledge-bases/{kb_id}/knowledge` | 知识条目列表 |
| GET | `/api/v1/knowledge-bases/knowledge/{knowledge_id}` | 知识条目详情 |
| DELETE | `/api/v1/knowledge-bases/knowledge/{knowledge_id}` | 删除知识条目 |

### 模型管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models` | 创建模型 |
| GET | `/api/v1/models` | 模型列表 |
| GET | `/api/v1/models/{model_id}` | 模型详情 |
| PATCH | `/api/v1/models/{model_id}` | 更新模型 |
| DELETE | `/api/v1/models/{model_id}` | 删除模型 |

## 关键文件清单

| 操作 | 文件 |
|------|------|
| 新建 | `app/schemas/knowledge.py` |
| 新建 | `app/schemas/model.py` |
| 新建 | `app/repositories/knowledge.py` |
| 新建 | `app/repositories/model.py` |
| 新建 | `app/services/knowledge_service.py` |
| 新建 | `app/services/model_service.py` |
| 新建 | `app/api/v1/knowledge.py` |
| 新建 | `app/api/v1/models.py` |
| 修改 | `app/schemas/__init__.py` - 添加导出 |
| 修改 | `app/api/v1/__init__.py` - 注册路由 |

## 设计决策

1. **完全复用现有模型** - `app/models/knowledge.py` 已定义完整的数据模型
2. **软删除策略** - 使用 `deleted_at` 字段而非物理删除
3. **租户隔离** - 所有操作自动添加 `tenant_id` 过滤
4. **异步优先** - 全面使用 `async/await` 模式
5. **分层架构** - API → Service → Repository → Model

## 验证步骤

### 1. 启动服务
```bash
uv run uvicorn app.main:app --reload
```

### 2. 创建嵌入模型
```bash
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "text-embedding-v4",
    "type": "Embedding",
    "source": "aliyun",
    "description": "DashScope 嵌入模型",
    "parameters": {
      "api_key": "your-api-key",
      "model_name": "text-embedding-v4",
      "dimensions": 1024
    },
    "is_default": true
  }'
```

### 3. 创建知识库
```bash
curl -X POST http://localhost:8000/api/v1/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "测试知识库",
    "description": "API 测试知识库",
    "embedding_model_id": "<model_id>",
    "summary_model_id": "<model_id>"
  }'
```

### 4. 查询知识库列表
```bash
curl http://localhost:8000/api/v1/knowledge-bases
```

### 5. 获取知识库详情
```bash
curl http://localhost:8000/api/v1/knowledge-bases/{kb_id}
```

## 依赖安装

```bash
# PostgreSQL + pgvector
uv add -E pgvector langchain-postgres pgvector

# LangChain 文档加载器
uv add -E loaders langchain-community langchain-text-splitters

# 文档处理（可选）
uv add python-magic
```
