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
| 知识库管理 | ❌ 无 | ✅ 已有 | **新增** 🔴 |
| 知识条目 | ❌ 无 | ✅ 已有 | **新增** 🔴 |
| 文档分块 | ❌ 无 | ✅ 已有 | **新增** 🟡 |
| 模型管理 | ❌ 无 | ✅ 已有 | **新增** 🔴 |
| 知识标签 | ❌ 无 | ✅ 已有 | **新增** 🟡 |
| 初始化系统 | ❌ 无 | ✅ 已有 | **新增** 🟢 |
| 网络搜索 | ❌ 无 | ✅ 已有 | **新增** 🟢 |

---

## ✅ 可以保留的接口

以下接口功能完整，无需修改：

| 模块 | 端点 | 文件路径 |
|------|------|----------|
| **认证** | `POST /auth/register` | `app/api/v1/auth.py` |
| | `POST /auth/login` | |
| | `POST /auth/refresh` | |
| | `GET /auth/me` | |
| **租户** | `GET /tenants` | `app/api/v1/tenants.py` |
| | `POST /tenants` | |
| | `GET /tenants/{id}` | |
| | `PATCH /tenants/{id}` | |
| | `DELETE /tenants/{id}` | |
| **会话** | `POST /sessions` | `app/api/v1/sessions.py` |
| | `GET /sessions` | |
| | `GET /sessions/{id}` | |
| | `PATCH /sessions/{id}` | |
| | `DELETE /sessions/{id}` | |
| | `POST /sessions/{id}/generate-title` | |
| **消息** | `GET /messages` | `app/api/v1/messages.py` |
| | `GET /messages/{id}` | |
| | `PATCH /messages/{id}` | |
| | `DELETE /messages/{id}` | |
| | `GET /messages/search` | |
| **聊天** | `POST /chat` | `app/api/v1/chat.py` |
| | `POST /chat/stream` | |
| | `GET /chat/history/{session_id}` | |
| **Agent** | `GET /agents` | `app/api/v1/agents.py` |
| | `POST /agents` | |
| | `GET /agents/{id}` | |
| | `PATCH /agents/{id}` | |
| | `DELETE /agents/{id}` | |
| | `GET /agents/stats` | |
| | `GET /agents/executions` | |
| **MCP** | `GET /mcp-services` | `app/api/v1/mcp_services.py` |
| | `POST /mcp-services` | |
| | `GET /mcp-services/{id}` | |
| | `PATCH /mcp-services/{id}` | |
| | `DELETE /mcp-services/{id}` | |
| **评估** | `POST /evaluation/run` | `app/api/v1/evaluation.py` |
| | `POST /evaluation/run/stream` | |
| | `GET /evaluation/results/{run_id}` | |

---

## ➕ 需要新增的接口

### 🔴 P0 - 核心功能（必须实现）

#### 1. 模型管理 `/models`

```python
# 文件: app/api/v1/models.py

POST   /models                              # 创建模型
GET    /models                              # 列表
GET    /models/{id}                          # 详情
PUT    /models/{id}                          # 更新
DELETE /models/{id}                          # 删除
GET    /models/providers                     # 获取模型厂商列表
```

**Schema 参考：**
- `type`: Embedding, Rerank, KnowledgeQA, VLLM, Chat
- `source`: local, remote, aliyun, zhipu, openai

---

#### 2. 知识库管理 `/knowledge-bases`

```python
# 文件: app/api/v1/knowledge_bases.py

POST   /knowledge-bases                      # 创建知识库
GET    /knowledge-bases                      # 列表 (分页)
GET    /knowledge-bases/{id}                  # 详情
PUT    /knowledge-bases/{id}                  # 更新
DELETE /knowledge-bases/{id}                  # 删除
GET    /knowledge-bases/{id}/hybrid-search    # 混合搜索
```

**Schema 参考：**
- `chunking_config`: 分块配置 (chunk_size, chunk_overlap, split_markers)
- `image_processing_config`: 多模态配置
- `embedding_model_id`: 嵌入模型关联
- `rerank_model_id`: 重排序模型关联
- `kb_type`: document, faq

---

#### 3. 知识条目管理 `/knowledge`

```python
# 文件: app/api/v1/knowledge.py

POST   /knowledge-bases/{id}/knowledge/file    # 从文件创建知识
POST   /knowledge-bases/{id}/knowledge/url     # 从URL创建知识
POST   /knowledge-bases/{id}/knowledge/manual  # 手工创建知识
GET    /knowledge-bases/{id}/knowledge        # 列表 (分页, 支持筛选)
GET    /knowledge/{id}                       # 详情
PUT    /knowledge/{id}                       # 更新
DELETE /knowledge/{id}                       # 删除
GET    /knowledge/{id}/download               # 下载原始文件
POST   /knowledge-search                     # 知识搜索 (无需session)
```

**Schema 参考：**
- `type`: file, url, text, faq
- `parse_status`: unprocessed, processing, completed, failed
- `enable_status`: enabled, disabled
- `file_name`, `file_type`, `file_size`, `file_path`

---

### 🟡 P1 - 重要功能（尽快实现）

#### 4. 知识标签管理 `/knowledge-bases/{id}/tags`

```python
# 文件: app/api/v1/knowledge_tags.py

GET    /knowledge-bases/{id}/tags            # 列表
POST   /knowledge-bases/{id}/tags            # 创建
PUT    /knowledge-bases/{id}/tags/{tag_id}   # 更新
DELETE /knowledge-bases/{id}/tags/{tag_id}   # 删除
```

---

#### 5. 文档分块管理 `/chunks`

```python
# 文件: app/api/v1/chunks.py

GET    /chunks/{knowledge_id}                 # 列出分块 (分页)
GET    /chunks/by-id/{id}                    # 通过ID获取分块
PUT    /chunks/{knowledge_id}/{id}           # 更新分块
DELETE /chunks/{knowledge_id}/{id}           # 删除分块
DELETE /chunks/{knowledge_id}               # 删除知识下所有分块
```

---

#### 6. 基于知识库的聊天

```python
# 修改: app/api/v1/chat.py

POST   /knowledge-chat/{session_id}          # 知识问答
POST   /agent-chat/{session_id}             # Agent问答
POST   /knowledge-search                     # 知识搜索
```

---

### 🟢 P2 - 增强功能（后续考虑）

#### 7. 租户配置 KV 存储

```python
# 文件: app/api/v1/tenant_config.py

GET    /tenants/kv/{key}                   # 获取配置值
PUT    /tenants/kv/{key}                   # 更新配置值
GET    /tenants/kv/agent-config            # 获取Agent配置
PUT    /tenants/kv/agent-config            # 更新Agent配置
GET    /tenants/kv/web-search-config       # 获取网络搜索配置
PUT    /tenants/kv/web-search-config       # 更新网络搜索配置
```

---

#### 8. 系统初始化

```python
# 文件: app/api/v1/initialization.py

POST   /initialization/initialize/{kbId}   # 初始化知识库
GET    /initialization/config/{kbId}       # 获取配置
PUT    /initialization/config/{kbId}       # 更新配置
POST   /initialization/embedding/test    # 测试嵌入模型
POST   /initialization/rerank/check       # 检查重排模型
POST   /initialization/ollama/status       # 检查Ollama状态
```

---

## 🔧 需要调整的接口

| 端点 | 当前状态 | 调整方案 | 优先级 |
|------|----------|----------|--------|
| `POST /sessions` | 创建会话 | 添加 `knowledge_base_ids`, `agent_config`, `context_config` 参数 | P0 |
| `POST /chat` | 聊天接口 | 添加知识库检索逻辑，参考 `/knowledge-chat/{session_id}` | P0 |
| `GET /tenants/{id}` | 租户详情 | 添加 retriever_engines, web_search_config 等字段 | P1 |
| `PATCH /tenants/{id}` | 更新租户 | 同上 | P1 |

---

## 📋 实施计划

### Phase 1: 基础设施（Week 1）
- [ ] 创建 `app/api/v1/models.py` - 模型管理
- [ ] 创建 `app/schemas/model.py` - 模型 Schema
- [ ] 创建 `app/repositories/model.py` - 模型 Repository
- [ ] 更新 `app/models/__init__.py` - 导出 Model

### Phase 2: 知识库（Week 1-2）
- [ ] 创建 `app/api/v1/knowledge_bases.py` - 知识库管理
- [ ] 创建 `app/api/v1/knowledge.py` - 知识条目管理
- [ ] 创建 `app/schemas/knowledge.py` - 知识库 Schema
- [ ] 创建 `app/services/knowledge_service.py` - 知识库服务

### Phase 3: 聊天集成（Week 2）
- [ ] 修改 `app/api/v1/chat.py` - 集成知识库检索
- [ ] 添加 `/knowledge-chat/{session_id}` 端点
- [ ] 添加混合搜索接口
- [ ] 更新会话创建逻辑

### Phase 4: 增强功能（Week 3）
- [ ] 创建 `app/api/v1/knowledge_tags.py` - 标签管理
- [ ] 创建 `app/api/v1/chunks.py` - 分块管理
- [ ] 添加租户 KV 配置接口
- [ ] 添加初始化系统接口

---

## 📝 备注

- 所有新增接口需要添加权限验证和租户隔离
- 遵循 RESTful 设计规范
- 统一响应格式（参考 WeKnora99）
- 流式响应使用 SSE (Server-Sent Events)
- 分页参数统一使用 `page` 和 `page_size`

---

*最后更新: 2025-01-31*
