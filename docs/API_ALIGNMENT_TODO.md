# Kiki 与 WeKnora99 API 对齐任务清单

> 创建时间: 2025-01-31
> 目标: 将 Kiki API 接口完全对齐 WeKnora99

---

## 📊 功能对比总览

| 功能模块 | Kiki 状态 | WeKnora99 | 对齐建议 |
|---------|----------|-----------|----------|
| 认证系统 | ✅ 已有 | ✅ 已有 | **保留** |
| 租户管理 | ✅ 已有 | ✅ 已有 | **调整** - 需添加 KV 配置 |
| 会话管理 | ✅ 已有 | ✅ 已有 | **保留** |
| 消息管理 | ✅ 已有 | ⚠️ 简化 | **保留** |
| 聊天接口 | ✅ 已有 | ✅ 已有 | **调整** - 集成知识库 |
| Agent 管理 | ✅ 已有 | ✅ 已有 | **保留** |
| MCP 服务 | ✅ 已有 | ✅ 已有 | **保留** |
| API Key 管理 | ✅ 已有 | ❌ 无 | **保留** |
| 评估系统 | ✅ 已有 | ✅ 已有 | **保留** |
| 知识库管理 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 知识条目 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 文档分块 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 模型管理 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 知识标签 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 初始化系统 | ✅ 已完成 | ✅ 已有 | **已完成** |
| 网络搜索 | ✅ 已完成 | ✅ 已有 | **已完成** |

---

## ✅ 已完成的功能

### 模型管理 (`app/api/v1/models.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `POST /models` | ✅ | 创建模型 |
| `GET /models` | ✅ | 模型列表 |
| `GET /models/{id}` | ✅ | 模型详情 |
| `PATCH /models/{id}` | ✅ | 更新模型 |
| `DELETE /models/{id}` | ✅ | 删除模型 |
| `GET /models/providers` | ✅ | 获取服务商列表 |

**支持的 ModelType:**
- `Embedding` - 嵌入模型
- `Rerank` - 重排序模型
- `KnowledgeQA` - 对话模型
- `Chat` - 聊天模型
- `VLLM` - 视觉语言模型

**支持的 Provider:**
- generic, openai, aliyun, zhipu, deepseek, jina, gemini, volcengine, hunyuan, siliconflow, moonshot

---

### 知识库管理 (`app/api/v1/knowledge.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `POST /knowledge-bases` | ✅ | 创建知识库 |
| `GET /knowledge-bases` | ✅ | 知识库列表 |
| `GET /knowledge-bases/{id}` | ✅ | 知识库详情 |
| `PUT /knowledge-bases/{id}` | ✅ | 更新知识库 |
| `DELETE /knowledge-bases/{id}` | ✅ | 删除知识库 |
| `POST /knowledge-bases/{id}/hybrid-search` | ✅ | 混合搜索 |
| `POST /knowledge-bases/copy` | ✅ | 拷贝知识库 |

**知识库配置:**
- `chunking_config`: 分块配置
- `embedding_model_id`: 嵌入模型
- `summary_model_id`: 摘要模型
- `rerank_model_id`: 重排序模型
- `vlm_config`: VLM 配置
- `image_processing_config`: 图像处理配置
- `cos_config`: 腾讯云存储配置

---

### 知识条目管理 (`app/api/v1/knowledge.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `POST /knowledge-bases/{id}/knowledge/file` | ✅ | 从文件创建 |
| `POST /knowledge-bases/{id}/knowledge/url` | ✅ | 从 URL 创建 |
| `POST /knowledge-bases/{id}/knowledge/manual` | ✅ | 手工创建 |
| `GET /knowledge-bases/{id}/knowledge` | ✅ | 知识列表 |
| `GET /knowledge/{id}` | ✅ | 知识详情 |
| `PUT /knowledge/{id}` | ✅ | 更新知识 |
| `DELETE /knowledge/{id}` | ✅ | 删除知识 |
| `GET /knowledge/{id}/download` | ✅ | 下载文件 |

---

### 网络搜索 (`app/api/v1/web_search.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `POST /web-search/search` | ✅ | 执行搜索 |
| `GET /web-search/providers` | ✅ | 获取提供商 |
| `POST /web-search/compress` | ✅ | RAG 压缩（预留） |

---

### 知识标签管理 (`app/api/v1/knowledge_tags.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /knowledge-bases/{id}/tags` | ✅ | 标签列表 |
| `POST /knowledge-bases/{id}/tags` | ✅ | 创建标签 |
| `PUT /knowledge-bases/{id}/tags/{tag_id}` | ✅ | 更新标签 |
| `DELETE /knowledge-bases/{id}/tags/{tag_id}` | ✅ | 删除标签 |

**标签配置:**
- `name`: 标签名称
- `color`: 标签颜色（十六进制）
- `sort_order`: 排序顺序
- `knowledge_count`: 关联知识数量
- `chunk_count`: 关联分块数量

---

### 系统初始化 (`app/api/v1/initialization.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /initialization/kb/{kb_id}/config` | ✅ | 获取知识库配置 |
| `PUT /initialization/kb/{kb_id}/config` | ✅ | 更新知识库配置 |
| `POST /initialization/kb/{kb_id}` | ✅ | 初始化知识库 |
| `GET /initialization/ollama/status` | ✅ | 检查 Ollama 状态 |
| `POST /initialization/models/embedding/test` | ✅ | 测试 Embedding 模型 |
| `POST /initialization/models/rerank/check` | ✅ | 检查 Rerank 模型 |
| `POST /initialization/models/remote/check` | ✅ | 检查远程模型连接 |

**初始化配置:**
- `llm`: LLM 模型配置
- `embedding`: Embedding 模型配置
- `rerank`: Rerank 模型配置
- `multimodal`: 多模态配置
- `document_splitting`: 文档分块配置

---

### 租户 KV 配置 (`app/api/v1/tenant_config.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /tenants/kv/{key}` | ✅ | 获取配置值 |
| `PUT /tenants/kv/{key}` | ✅ | 更新配置值 |
| `GET /tenants/kv/agent-config` | ✅ | 获取 Agent 配置 |
| `PUT /tenants/kv/agent-config` | ✅ | 更新 Agent 配置 |
| `GET /tenants/kv/web-search-config` | ✅ | 获取网络搜索配置 |
| `PUT /tenants/kv/web-search-config` | ✅ | 更新网络搜索配置 |

**配置存储:**
- 使用 PostgreSQL JSONB 存储 KV 配置
- 支持任意键值对存储
- 内置 Agent 和网络搜索配置快捷接口

---

### 文档分块管理 (`app/api/v1/chunks.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /chunks/{knowledge_id}` | ✅ | 获取知识的分块列表 |
| `GET /chunks/by-id/{id}` | ✅ | 通过 ID 获取分块 |
| `PUT /chunks/{knowledge_id}/{id}` | ✅ | 更新分块 |
| `DELETE /chunks/{knowledge_id}/{id}` | ✅ | 删除分块 |
| `DELETE /chunks/{knowledge_id}` | ✅ | 删除知识下所有分块 |
| `DELETE /chunks/by-id/{id}/questions` | ✅ | 删除生成的问题 |

---

### 系统信息 (`app/api/v1/system.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /system/info` | ✅ | 获取系统信息 |
| `GET /system/minio/buckets` | ✅ | 列出存储桶（预留） |

---

### Ollama 模型管理 (`app/api/v1/initialization.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /initialization/ollama/status` | ✅ | 检查 Ollama 状态 |
| `GET /initialization/ollama/models` | ✅ | 列出 Ollama 模型 |
| `POST /initialization/ollama/models/check` | ✅ | 检查模型是否存在 |
| `POST /initialization/ollama/models/download` | ✅ | 下载 Ollama 模型 |
| `GET /initialization/ollama/download/progress/{task_id}` | ✅ | 获取下载进度 |
| `GET /initialization/ollama/download/tasks` | ✅ | 下载任务列表 |

---

### 知识库 FAQ 管理 (`app/api/v1/knowledge_faq.py`)

| 端点 | 状态 | 说明 |
|------|------|------|
| `GET /knowledge-bases/{id}/faq/entries` | ✅ | FAQ 条目列表 |
| `GET /knowledge-bases/{id}/faq/entries/export` | ✅ | 导出 FAQ 为 CSV |
| `GET /knowledge-bases/{id}/faq/entries/{entry_id}` | ✅ | FAQ 条目详情 |
| `POST /knowledge-bases/{id}/faq/entries` | ✅ | 批量创建/更新 FAQ |
| `POST /knowledge-bases/{id}/faq/entry` | ✅ | 创建单个 FAQ |
| `PUT /knowledge-bases/{id}/faq/entries/{entry_id}` | ✅ | 更新 FAQ 条目 |
| `POST /knowledge-bases/{id}/faq/entries/{entry_id}/similar-questions` | ✅ | 添加相似问题 |
| `PUT /knowledge-bases/{id}/faq/entries/fields` | ✅ | 批量更新字段 |
| `PUT /knowledge-bases/{id}/faq/entries/tags` | ✅ | 批量更新标签 |
| `DELETE /knowledge-bases/{id}/faq/entries` | ✅ | 删除 FAQ 条目 |
| `POST /knowledge-bases/{id}/faq/search` | ✅ | 搜索 FAQ |
| `GET /faq/import/progress/{task_id}` | ✅ | 导入进度查询 |

---

## 📋 实施进度

### Phase 1: 基础设施 ✅
- [x] 创建 `app/api/v1/models.py` - 模型管理
- [x] 创建 `app/schemas/model.py` - 模型 Schema
- [x] 创建 `app/repositories/model.py` - 模型 Repository
- [x] 更新 `app/models/__init__.py` - 导出 Model

### Phase 2: 知识库 ✅
- [x] 创建 `app/api/v1/knowledge.py` - 知识库管理
- [x] 创建 `app/schemas/knowledge.py` - 知识库 Schema
- [x] 创建 `app/services/knowledge_service.py` - 知识库服务
- [x] 实现知识库拷贝功能

### Phase 3: 网络搜索 ✅
- [x] 创建 `app/api/v1/web_search.py` - 网络搜索
- [x] 创建 `app/services/web_search.py` - 搜索服务
- [x] 创建 `app/schemas/web_search.py` - 搜索 Schema

### Phase 4: 增强功能 ✅
- [x] 创建 `app/api/v1/knowledge_tags.py` - 标签管理
- [x] 创建 `app/schemas/knowledge.py` - 标签 Schema
- [x] 创建 `app/repositories/tag.py` - 标签 Repository
- [x] 创建 `app/api/v1/initialization.py` - 系统初始化
- [x] 创建 `app/schemas/initialization.py` - 初始化 Schema
- [x] 创建 `app/services/initialization_service.py` - 初始化服务
- [x] 创建 `app/api/v1/tenant_config.py` - 租户 KV 配置
- [x] 添加 `kv_config` 字段到 Tenant 模型

---

## 📝 备注

- 所有新增接口需要添加权限验证和租户隔离
- 遵循 RESTful 设计规范
- 统一响应格式（参考 WeKnora99）
- 流式响应使用 SSE (Server-Sent Events)
- 分页参数统一使用 `page` 和 `size`

---

*最后更新: 2025-01-31*
