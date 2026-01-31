# Kiki vs WeKnora 功能对比分析

> 分析日期: 2025-01-31
> 目的: 对齐 WeKnora99 业务逻辑，识别差异，规划迁移

## 一、项目概览对比

| 维度 | Kiki (Python) | WeKnora (Go) |
|------|---------------|--------------|
| **语言** | Python 3.13+ | Go 1.24 |
| **Web 框架** | FastAPI + Uvicorn | Gin |
| **ORM** | SQLModel | GORM |
| **数据库** | PostgreSQL | PostgreSQL |
| **向量数据库** | - | Qdrant |
| **图数据库** | - | Neo4j |
| **搜索引擎** | - | Elasticsearch |
| **Agent 框架** | LangGraph | 自研 |
| **前端** | Vue 3 | Vue 3 |

## 二、Kiki 独有的功能模块

### 2.1 API 层面

| 模块 | 说明 | API 端点 |
|------|------|----------|
| **评估系统** | Agent 性能评估、数据集管理 | `POST /api/v1/evaluation/run` |
| | | `GET /api/v1/evaluation/results` |
| | | `GET /api/v1/evaluation/datasets` |
| **工具管理** | 列出/获取工具详情 | `GET /api/v1/tools/` |
| | | `GET /api/v1/tools/{tool_name}` |
| **Agent 执行历史** | 追踪 Agent 执行记录 | `GET /api/v1/agents/executions` |
| **Agent 统计** | Agent 使用统计信息 | `GET /api/v1/agents/stats` |
| **聊天上下文管理** | 上下文统计/清除 | `GET /api/v1/chat/context/{session_id}/stats` |
| | | `DELETE /api/v1/chat/context/{session_id}` |
| **会话标题生成** | AI 自动生成会话标题 | `POST /api/v1/sessions/{id}/generate-title` |
| **搜索提供商** | 获取可用搜索提供商 | `GET /api/v1/chat/search/providers` |
| **API Key 统计** | API Key 使用统计 | `GET /api/v1/api-keys/stats/me` |

### 2.2 代码模块层面

| 模块路径 | 说明 |
|----------|------|
| `app/evaluation/` | 评估模块 |
| `app/rate_limit/` | Token Bucket 限流 |
| `app/llm/` | 多提供商 LLM 管理 |
| `app/observability/` | structlog + Prometheus |
| `app/auth/` | 完整认证授权模块 |
| `app/repositories/memory.py` | 长期记忆仓储 |
| `app/repositories/thread.py` | LangGraph 线程仓储 |

### 2.3 数据模型层面

| 模型 | 说明 |
|------|------|
| `AuthToken` | 认证令牌模型 |
| `Thread` | LangGraph 线程模型 |
| `Memory` | 长期记忆模型 |

## 三、WeKnora 独有的功能模块

### 3.1 API 层面

| 模块 | 说明 | Handler 文件 | Kiki 对齐状态 |
|------|------|--------------|---------------|
| **FAQ 系统** | 常见问答管理 | `internal/handler/faq.go` | ✅ 已对齐 |
| **标签系统** | 内容分类标签 | `internal/handler/tag.go` | ✅ 已对齐 (knowledge_tags) |
| **网络搜索** | 实时网页搜索 | `internal/handler/web_search.go` | ✅ 已对齐 |
| **文档解析** | 文档上传和解析 | `internal/handler/chunk.go` | ✅ 已对齐 |

### 3.2 服务模块

| 模块路径 | 说明 |
|----------|------|
| `docreader/` | Python 文档解析服务 |
| | - PDF (PyMuPDF) |
| | - Word (python-docx) |
| | - Excel (openpyxl) |
| | - PPT (python-pptx) |
| | - HTML/Markdown |
| `client/` | Go 客户端 SDK |

### 3.3 基础设施

| 组件 | 说明 |
|------|------|
| **Qdrant** | 向量数据库 |
| **Neo4j** | 图数据库 (图 RAG) |
| **Elasticsearch** | 全文搜索引擎 |
| **MinIO/COS** | 对象存储 |

## 四、功能对齐建议

### 4.1 Kiki 保留的功能

| 功能 | 保留理由 | 优先级 |
|------|----------|--------|
| 评估系统 | Agent 质量保证很重要 | 高 |
| 限流模块 | 生产环境必备 | 高 |
| LLM 服务层 | 多提供商支持是优势 | 高 |
| 可观测性模块 | 监控和日志是生产必需 | 高 |
| 工具管理 API | 与 MCP 工具系统配合 | 中 |
| Agent 执行历史 | 调试和分析 | 中 |
| 会话标题生成 | 用户体验功能 | 低 |
| 聊天上下文管理 | LangGraph 状态管理需要 | 高 |

### 4.2 从 WeKnora 迁移的功能

| 功能 | 优先级 | 复杂度 | 状态 |
|------|--------|--------|------|
| **FAQ 系统** | 高 | 中 | ✅ 已对齐 |
| **文档解析服务** | 高 | 中 | ✅ 已对齐 (LangChain) |
| **网络搜索** | 高 | 中 | ✅ 已对齐 |
| **标签系统** | 中 | 低 | ✅ 已对齐 (knowledge_tags) |
| **模型管理** | 高 | 中 | ✅ 已对齐 |
| **知识库管理** | 高 | 中 | ✅ 已对齐 |
| **文档分块管理** | 中 | 低 | ✅ 已对齐 |
| **系统初始化** | 中 | 中 | ✅ 已对齐 |
| **租户 KV 配置** | 中 | 低 | ✅ 已对齐 |
| **Ollama 模型管理** | 中 | 中 | ✅ 已对齐 |
| **知识库 FAQ** | 高 | 中 | ✅ 已对齐 |
| **向量数据库集成** | 高 | 中 | ✅ 已对齐 (Qdrant + Pinecone) |
| **搜索引擎集成** | 中 | 中 | ✅ 已对齐 (LangChain Elasticsearch) |
| **知识库混合搜索** | 高 | 中 | ✅ 已对齐 (RRF + Rerank) |
| **Agent 复制** | 中 | 低 | ✅ 已对齐 |
| **会话停止** | 中 | 低 | ✅ 已对齐 |
| **知识库复制** | 中 | 中 | ✅ 已对齐 (异步任务) |
| **租户搜索** | 中 | 低 | ✅ 已对齐 |
| **系统信息增强** | 中 | 低 | ✅ 已对齐 |
| **独立知识搜索** | 高 | 中 | ✅ 已对齐 (LangChain) |
| **BM25 检索** | 中 | 低 | ✅ 已对齐 (LangChain) |
| **集成检索器** | 中 | 低 | ✅ 已对齐 (LangChain) |
| **Agent 占位符** | 中 | 低 | ✅ 已对齐 |
| **消息分页加载** | 低 | 低 | ✅ 已对齐 |
| **FAQ 导出** | 低 | 低 | ✅ 已对齐 |
| **继续接收流** | 中 | 低 | ✅ 已对齐 |

## 五、技术架构差异

### 5.1 分层架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kiki (FastAPI)                           │
├─────────────────────────────────────────────────────────────────┤
│  API 层 (app/api/)    → 路由、中间件                            │
│  服务层 (app/services/) → 业务逻辑                              │
│  仓储层 (app/repositories/) → 数据访问                          │
│  模型层 (app/models/)    → SQLModel 定义                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        WeKnora (Gin)                            │
├─────────────────────────────────────────────────────────────────┤
│  Handler (internal/handler/) → HTTP 处理器                      │
│  Service (内嵌在 handler)     → 业务逻辑                         │
│  Models (internal/models/)   → GORM 模型                        │
│  Types (internal/types/)     → 类型定义                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Agent 系统对比

| 维度 | Kiki | WeKnora |
|------|------|---------|
| 框架 | LangGraph | 自研 |
| 状态管理 | Checkpointer 持久化 | 内存/数据库 |
| 工具系统 | LangChain + MCP | 自研 + MCP |
| 记忆系统 | short_term + long_term | 简单记忆 |

## 六、迁移路线图

### Phase 1: 基础设施 (高优先级)

- [x] **网络搜索功能** - ✅ 已完成
  - `app/services/web_search.py` - 搜索服务
  - `app/services/web_search_providers.py` - 提供商抽象
  - `app/api/v1/web_search.py` - 搜索 API
  - 支持 DuckDuckGo、Tavily、Google、Bing
- [x] **文档解析服务** - ✅ 已完成 (LangChain)
  - `app/services/document_loaders.py` - 文档加载器
  - `app/services/document_splitter.py` - 文档分块
  - `app/services/document_service.py` - 文档服务
  - `app/api/v1/documents.py` - 文档 API
  - 支持 PDF、Word、Excel、PPT、Markdown、网页
- [ ] 集成 Qdrant 向量数据库
- [ ] 集成 Elasticsearch 搜索引擎

### Phase 2: 核心功能 (高优先级)

- [x] **FAQ 系统** - ✅ 已完成
  - `app/models/faq.py` - FAQ 数据模型
  - `app/services/faq_service.py` - FAQ 服务
  - `app/repositories/faq.py` - FAQ 仓储
  - `app/api/v1/faq.py` - FAQ API
- [x] **标签系统** - ✅ 已完成
  - `app/api/v1/knowledge_tags.py` - 标签 API
  - `app/schemas/knowledge.py` - Tag 相关 Schema
  - `app/repositories/tag.py` - Tag 仓储
- [x] **模型管理** - ✅ 已完成
  - `app/api/v1/models.py` - 模型 API
  - `app/repositories/model.py` - 模型仓储
- [x] **知识库管理** - ✅ 已完成
  - `app/api/v1/knowledge.py` - 知识库 API
  - `app/services/knowledge_service.py` - 知识库服务

### Phase 3: 高级功能 (中优先级)

- [ ] 图数据库集成 (Neo4j)
- [ ] 图 RAG 实现
- [ ] 高级检索策略

## 七、数据模型对齐状态

| 模型 | Kiki | WeKnora | 状态 |
|------|------|---------|------|
| User | ✅ | ✅ | ✅ 对齐 |
| Tenant | ✅ | ✅ | ✅ 对齐 |
| Session | ✅ | ✅ | ✅ 对齐 |
| Message | ✅ | ✅ | ✅ 对齐 |
| Agent/CustomAgent | ✅ | ✅ | ✅ 对齐 |
| KnowledgeBase | ✅ | ✅ | ✅ 对齐 |
| Knowledge | ✅ | ✅ | ✅ 对齐 |
| Chunk | ✅ | ✅ | ✅ 对齐 |
| Model | ✅ | ✅ | ✅ 对齐 |
| ApiKey | ✅ | ✅ | ✅ 对齐 |
| MCPService | ✅ | ✅ | ✅ 对齐 |
| **FAQ** | ✅ | ✅ | ✅ 已对齐 |
| **Tag** | ✅ | ✅ | ✅ 已对齐 |

## 八、网络搜索功能对齐详情

### 8.1 已实现的功能

| 功能 | Kiki 实现 | WeKnora 参考 |
|------|----------|--------------|
| **多提供商支持** | `app/services/web_search_providers.py` | `internal/application/service/web_search/` |
| **搜索服务** | `WebSearchService` | `WebSearchService` |
| **提供商注册表** | `WebSearchProviderRegistry` | `Registry` |
| **黑名单过滤** | `filter_blacklist()` | `filterBlacklist()` |
| **API 端点** | `/api/v1/web-search/*` | `/web-search/*` |

### 8.2 支持的搜索引擎

| 提供商 | 免费版 | API Key | Kiki 状态 |
|--------|--------|---------|-----------|
| **DuckDuckGo** | ✅ | ❌ | ✅ 完全支持 |
| **Tavily** | ❌ | ✅ | ✅ 完全支持 |
| **Google** | ❌ | ✅ | ✅ 预留接口 |
| **Bing** | ❌ | ✅ | ✅ 预留接口 |

### 8.3 API 端点对比

| 功能 | Kiki 端点 | WeKnora 端点 |
|------|-----------|--------------|
| 执行搜索 | `POST /api/v1/web-search/search` | `POST /web-search/search` |
| 获取提供商 | `GET /api/v1/web-search/providers` | `GET /web-search/providers` |
| RAG 压缩 | `POST /api/v1/web-search/compress` | `POST /web-search/compress` |

### 8.4 配置对比

| 配置项 | Kiki | WeKnora |
|--------|------|---------|
| provider | ✅ | ✅ |
| max_results | ✅ | ✅ |
| include_date | ✅ | ✅ |
| blacklist | ✅ | ✅ |
| compression_method | ✅ (预留) | ✅ |
| embedding_model_id | ✅ (预留) | ✅ |

### 8.5 安装依赖

```bash
# 安装网络搜索依赖
uv add -E websearch duckduckgo-search tavily-python

# 或使用可选依赖组
uv sync --extra websearch
```

### 8.6 环境变量配置

```bash
# Tavily 搜索 API
TAVILY_API_KEY=your_tavily_api_key

# Google Custom Search (可选)
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Bing Search (可选)
BING_SEARCH_API_KEY=your_bing_api_key
```

## 九、文档处理功能对齐详情

### 9.1 技术选型

| 方案 | 选择 | 理由 |
|------|------|------|
| WeKnora docreader | ❌ 不使用 | gRPC 服务，增加部署复杂度 |
| LangChain 生态 | ✅ 使用 | 原生 Python 集成，社区活跃 |

### 9.2 已实现的功能

| 功能 | Kiki 实现 | LangChain 组件 |
|------|----------|----------------|
| **PDF 解析** | `PDFLoader` | PyMuPDFLoader |
| **Word 解析** | `WordLoader` | python-docx |
| **Excel 解析** | `ExcelLoader` | openpyxl |
| **PPT 解析** | `PPTLoader` | python-pptx |
| **Markdown 解析** | `MarkdownLoaderWrapper` | UnstructuredMarkdownLoader |
| **网页解析** | `WebLoaderWrapper` | httpx + BeautifulSoup4 |
| **文本分块** | `DocumentSplitter` | RecursiveCharacterTextSplitter |

### 9.3 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/documents/parse` | POST | 解析文档（JSON） |
| `/documents/parse/upload` | POST | 解析上传文件 |
| `/documents/parse/url` | POST | 解析网页内容 |
| `/documents/formats` | GET | 获取支持格式 |
| `/documents/parse/batch` | POST | 批量解析 |

### 9.4 支持的文档格式

| 格式 | 扩展名 | 状态 |
|------|--------|------|
| PDF | .pdf | ✅ |
| Word | .doc, .docx | ✅ |
| Excel | .xls, .xlsx, .xlsm | ✅ |
| PowerPoint | .ppt, .pptx | ✅ |
| Text | .txt | ✅ |
| Markdown | .md, .markdown | ✅ |
| Web | URL | ✅ |

### 9.5 安装依赖

```bash
# 文档处理依赖已包含在主依赖中
uv sync

# 包含的包：
# - pypdf, pymupdf (PDF)
# - python-docx (Word)
# - openpyxl (Excel)
# - python-pptx (PowerPoint)
# - beautifulsoup4, lxml (网页)
# - pillow (图片)
# - langchain-text-splitters (分块)
```

## 十、FAQ 功能对齐详情

### 10.1 已实现的功能

| 功能 | Kiki 实现 | 说明 |
|------|----------|------|
| **FAQ 数据模型** | `app/models/faq.py` | SQLModel，支持多租户 |
| **FAQ 仓储** | `app/repositories/faq.py` | 数据访问层 |
| **FAQ 服务** | `app/services/faq_service.py` | 业务逻辑层 |
| **FAQ API** | `app/api/v1/faq.py` | RESTful API |

### 10.2 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/faq` | POST | 创建 FAQ |
| `/faq` | GET | 列出 FAQ（支持筛选） |
| `/faq/{faq_id}` | GET | 获取 FAQ 详情 |
| `/faq/{faq_id}` | PATCH | 更新 FAQ |
| `/faq/{faq_id}` | DELETE | 删除 FAQ |
| `/faq/search` | POST | 搜索 FAQ |
| `/faq/{faq_id}/feedback` | POST | 提交反馈 |
| `/faq/stats/overview` | GET | 获取统计 |
| `/faq/bulk/status` | POST | 批量更新状态 |
| `/faq/public/published` | GET | 公开 FAQ 列表 |

### 10.3 FAQ 数据模型

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| tenant_id | int | 租户 ID |
| question | str | 问题 |
| answer | str | 答案 |
| category | enum | 分类 |
| status | enum | 状态 |
| locale | str | 语言 |
| order | int | 排序 |
| view_count | int | 浏览次数 |
| helpful_count | int | 有用反馈数 |

## 十一、总结

### 11.1 核心结论

1. **数据模型已高度对齐** - 主要业务实体基本一致
2. **Kiki 优势** - LangGraph 集成、评估系统、Python 生态
3. **核心功能已完成对齐** - 网络搜索、FAQ、文档解析
4. **待补充功能** - 标签系统、向量数据库集成

### 11.2 对齐进度

| 功能 | 状态 | 进度 |
|------|------|------|
| 网络搜索 | ✅ 已完成 | 100% |
| FAQ 系统 | ✅ 已完成 | 100% |
| 文档解析 | ✅ 已完成 | 100% |
| 标签系统 | ✅ 已完成 | 100% |
| 模型管理 | ✅ 已完成 | 100% |
| 知识库管理 | ✅ 已完成 | 100% |
| 文档分块管理 | ✅ 已完成 | 100% |
| 系统初始化 | ✅ 已完成 | 100% |
| 租户 KV 配置 | ✅ 已完成 | 100% |
| Ollama 模型管理 | ✅ 已完成 | 100% |
| 知识库 FAQ | ✅ 已完成 | 100% |
| 向量数据库 | ✅ 已完成 | 100% (Qdrant + Pinecone + Memory) |

### 11.3 技术栈对比

| 模块 | WeKnora | Kiki | 选择 |
|------|---------|------|------|
| 网络搜索 | 自研 | DuckDuckGo/Tavily | ✅ 标准 |
| 文档解析 | gRPC 服务 | LangChain | ✅ 更简洁 |
| FAQ | 自研 | SQLModel + Service | ✅ 对齐 |
| Agent | 自研 | LangGraph | ✅ 更强 |

### 11.4 推荐行动

1. **保留 Kiki 特有优势** - 评估系统、可观测性、LLM 服务层
2. **使用 LangChain 生态** - 文档处理、向量存储
3. **已完成 API 对齐** - 标签系统、模型管理、知识库等核心 API 已完成
4. **向量数据库已集成** - 支持 Qdrant、Pinecone、Memory

## 十二、Elasticsearch 搜索引擎集成详情

### 12.1 技术选型

| 方案 | 选择 | 理由 |
|------|------|------|
| 自研封装 | ❌ 不使用 | 维护成本高 |
| LangChain 集成 | ✅ 使用 | 官方支持，持续更新 |

### 12.2 已实现的功能

| 功能 | Kiki 实现 | LangChain 组件 |
|------|----------|----------------|
| **向量存储** | `ElasticsearchVectorStore` | `ElasticsearchStore` |
| **密集向量搜索** | DenseVectorStrategy | 内置 embedding |
| **稀疏向量搜索** | SparseVectorStrategy | ELSER 模型 |
| **混合搜索** | HybridSearchConfig | dense + sparse |
| **检索器** | `ElasticsearchRetriever` | `BaseRetriever` |

### 12.3 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/elasticsearch/health` | GET | 健康检查 |
| `/elasticsearch/indices` | POST | 创建索引 |
| `/elasticsearch/indices` | GET | 列出索引 |
| `/elasticsearch/indices/{index}` | DELETE | 删除索引 |
| `/elasticsearch/documents` | POST | 索引文档 |
| `/elasticsearch/search` | POST | 搜索 |
| `/elasticsearch/search/hybrid` | POST | 混合搜索 |
| `/elasticsearch/search/raw` | POST | 原始查询 |
| `/elasticsearch/analyze` | POST | 文本分析 |

### 12.4 安装依赖

```bash
# Elasticsearch 集成
uv add -E elasticsearch "elasticsearch[async]>=8.0.0" langchain-elasticsearch
```

### 12.5 配置

```bash
# Elasticsearch 连接
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_API_KEY=your_api_key  # 可选，云端部署
```

## 十三、混合搜索功能对齐详情

### 13.1 已实现的功能

| 功能 | Kiki 实现 | WeKnora 参考 |
|------|----------|--------------|
| **向量搜索** | `VectorSearcher` | VectorRetriever |
| **关键词搜索** | `KeywordSearcher` | KeywordsRetriever |
| **RRF 融合** | `RRFCombiner` (k=60) | RRF fusion |
| **重排序** | `RerankerService` | Cohere rerank |
| **并行搜索** | `asyncio.gather` | goroutine |

### 13.2 支持的重排序器

| 重排序器 | 说明 | 来源 |
|----------|------|------|
| **Cohere** | cohere-rerank-v3 | API |
| **Jina** | jina-reranker-v2 | API |
| **Local** | 基于词汇相似度 | 本地降级 |

### 13.3 API 端点

```
POST /knowledge-bases/{kb_id}/hybrid-search
```

**请求参数：**
```json
{
  "query_text": "搜索查询",
  "vector_threshold": 0.5,
  "keyword_threshold": 0.3,
  "match_count": 5,
  "enable_rerank": true,
  "rerank_model_id": "cohere-rerank-v3",
  "top_k": 20
}
```

## 十四、Agent 复制功能对齐详情

### 14.1 API 端点

```
POST /agents/{agent_id}/copy
POST /agents/batch/copy
```

### 14.2 复制选项

| 参数 | 说明 |
|------|------|
| `name` | 新 Agent 名称 |
| `description` | 新 Agent 描述 |
| `copy_config` | 是否复制配置 |
| `copy_tools` | 是否复制工具关联 |
| `copy_knowledge` | 是否复制知识库关联 |

## 十五、会话停止功能对齐详情

### 15.1 API 端点

```
POST /sessions/{session_id}/stop
GET  /sessions/{session_id}/state
```

### 15.2 实现方式

| 组件 | 说明 |
|------|------|
| `CancellationToken` | Redis 存储停止标记 |
| `SessionStateManager` | 会话状态管理 |
| `CancellableStreamProcessor` | 可取消流式处理器 |

## 十二、向量数据库集成详情

```bash
# Elasticsearch 连接
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_API_KEY=your_api_key  # 可选，云端部署
```

## 十三、向量数据库集成详情

### 12.1 支持的向量存储

| 存储 | 类型 | 说明 |
|------|------|------|
| **Memory** | 内存 | 开发/测试用，无需额外服务 |
| **Qdrant** | 本地/云端 | 高性能向量数据库，与 WeKnora 对齐 |
| **Pinecone** | 云端 | 托管向量搜索服务 |

### 12.2 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/vectors/index` | POST | 索引文档 |
| `/vectors/search` | POST | 向量搜索 |
| `/vectors/search/hybrid` | POST | 混合搜索（向量+关键词） |
| `/vectors/collections` | POST | 创建集合 |
| `/vectors/{collection}` | DELETE | 删除集合 |
| `/vectors/stats` | GET | 获取统计 |
| `/vectors/health` | GET | 健康检查 |

### 12.3 安装依赖

```bash
# Qdrant（推荐，与 WeKnora 对齐）
uv add -E qdrant qdrant-client

# Pinecone（可选，云端服务）
uv add -E pinecone pinecone
```

### 12.4 配置

```bash
# settings.py 或环境变量
VECTOR_STORE_TYPE=qdrant  # memory, qdrant, pinecone
QDRANT_HOST=localhost:6334
QDRANT_API_KEY=your_key  # 云端部署需要
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east1-aws
```

---

## 十六、知识库复制功能详情

### 16.1 API 端点

```
POST /knowledge-bases/copy
GET  /knowledge-bases/copy/progress/{task_id}
```

### 16.2 复制流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 启动复制任务                                          │
│     POST /knowledge-bases/copy                             │
│     → 返回 task_id                                         │
├─────────────────────────────────────────────────────────────┤
│  2. 后台异步执行                                          │
│     - 复制标签                                             │
│     - 复制知识条目                                         │
│     - 复制文档分块                                         │
│     - 更新进度到 Redis                                     │
├─────────────────────────────────────────────────────────────┤
│  3. 查询进度                                              │
│     GET /knowledge-bases/copy/progress/{task_id}           │
│     → 返回进度百分比、已完成计数                            │
└─────────────────────────────────────────────────────────────┘
```

### 16.3 复制选项

| 参数 | 说明 |
|------|------|
| `task_id` | 可选，不传自动生成 UUID |
| `source_id` | 源知识库 ID（必填） |
| `target_id` | 目标知识库 ID，为空时创建新知识库 |
| `target_name` | 新知识库名称 |
| `copy_tags` | 是否复制标签 |
| `copy_knowledges` | 是否复制知识条目 |
| `copy_chunks` | 是否复制文档分块 |

## 十七、租户搜索功能详情

### 17.1 API 端点

```
GET /tenants/search?keyword=xxx&page=1&size=20
```

### 17.2 搜索字段

| 字段 | 说明 |
|------|------|
| `name` | 租户名称 |
| `description` | 租户描述 |
| `business` | 业务类型 |

### 17.3 响应格式

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20
}
```

## 十八、独立知识搜索功能详情（LangChain）

### 18.1 API 端点

```
POST /knowledge-bases/knowledge-search
```

### 18.2 支持的检索器

| 类型 | LangChain 组件 | 说明 |
|------|----------------|------|
| `vector` | `VectorStoreRetriever` | 向量相似度搜索 |
| `bm25` | `BM25Retriever` | BM25 关键词搜索 |
| `ensemble` | `EnsembleRetriever` | 融合多个检索器 |
| `conversational` | `ConversationalRetrievalChain` | 对话式检索+LLM |

### 18.3 代码结构

```
app/retrievers/
├── __init__.py           # 统一导出
├── base.py               # 基础检索器
├── elasticsearch.py      # ES 检索器
├── bm25.py               # BM25 检索器（LangChain）
├── ensemble.py           # 集成检索器（RRF）
└── conversational.py     # 对话式检索器（LangChain）
```

### 18.4 使用示例

```python
from langchain_community.retrievers import BM25Retriever
from app.retrievers.conversational import ConversationalRetriever

# 直接使用 LangChain
retriever = BM25Retriever.from_documents(docs)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
result = qa_chain.invoke({"question": "查询", "chat_history": []})
```

## 十九、系统信息功能详情

### 18.1 API 端点

| 端点 | 说明 |
|------|------|
| `GET /system/info` | 完整系统信息 |
| `GET /system/health` | 健康检查 |
| `GET /system/engines` | 引擎状态 |
| `GET /system/storage` | 存储状态 |

### 18.2 系统信息内容

```json
{
  "version": {
    "version": "0.1.0",
    "commit_id": "abc123",
    "build_time": "2025-01-31T00:00:00Z",
    "python_version": "3.13.0"
  },
  "engines": {
    "vector": "qdrant",
    "search": "elasticsearch",
    "graph": null
  },
  "storage": {
    "type": "minio",
    "buckets": ["knowledge", "uploads", "cache"]
  },
  "environment": "development"
}
```

---

## 十二、API 对齐完成清单

### 12.1 已完成的对齐模块

| 模块 | API 文件 | 对齐状态 |
|------|----------|----------|
| 认证系统 | `app/api/v1/auth.py` | ✅ |
| 租户管理 | `app/api/v1/tenants.py` | ✅ |
| 会话管理 | `app/api/v1/sessions.py` | ✅ |
| 消息管理 | `app/api/v1/messages.py` | ✅ |
| 聊天接口 | `app/api/v1/chat.py` | ✅ |
| Agent 管理 | `app/api/v1/agents.py` | ✅ |
| MCP 服务 | `app/api/v1/mcp_services.py` | ✅ |
| API Key 管理 | `app/api/v1/api_keys.py` | ✅ |
| FAQ 管理 | `app/api/v1/faq.py` | ✅ |
| 网络搜索 | `app/api/v1/web_search.py` | ✅ |
| 模型管理 | `app/api/v1/models.py` | ✅ |
| 知识库管理 | `app/api/v1/knowledge.py` | ✅ |
| 知识标签 | `app/api/v1/knowledge_tags.py` | ✅ |
| 文档分块 | `app/api/v1/chunks.py` | ✅ |
| 系统初始化 | `app/api/v1/initialization.py` | ✅ |
| 租户配置 | `app/api/v1/tenant_config.py` | ✅ |
| 系统信息 | `app/api/v1/system.py` | ✅ |
| 知识库 FAQ | `app/api/v1/knowledge_faq.py` | ✅ |
| 向量存储 | `app/api/v1/vectors.py` | ✅ (memory store) |
| Elasticsearch | `app/api/v1/elasticsearch.py` | ✅ |
| 文档解析 | `app/api/v1/documents.py` | ✅ |

### 12.2 本轮新增对齐

| 模块 | 功能 | API |
|------|------|-----|
| 混合搜索 | 向量+关键词+RRF融合 | `POST /knowledge-bases/{id}/hybrid-search` |
| Agent 复制 | 快速创建相似 Agent | `POST /agents/{id}/copy` |
| 会话停止 | 中断流式响应 | `POST /sessions/{id}/stop` |
| 知识库复制 | 异步复制大知识库 | `POST /knowledge-bases/copy` |
| 租户搜索 | 分页搜索租户 | `GET /tenants/search` |
| 系统信息 | 版本/引擎/存储状态 | `GET /system/info` |

---

*文档维护: 本文档应随着对齐进度持续更新*

## 二十、Agent 占位符功能详情

### 20.1 API 端点

```
GET  /agents/placeholders
POST /agents/placeholders
PUT  /agents/placeholders/{id}
DELETE /agents/placeholders/{id}
GET  /agents/{agent_id}/placeholders
POST /agents/preview
```

### 20.2 支持的占位符格式

| 格式 | 说明 | 示例 |
|------|------|------|
| `{{name}}` | Jinja2 格式 | `{{company_name}}` |
| `${name}` | 简单格式 | `${user_name}` |

### 20.3 变量类型

| 类型 | 验证规则 |
|------|----------|
| `string` | 无 |
| `email` | 正则 `^[^@]+@[^@]+\.[^@]+$` |
| `url` | 正则 `^https?://` |
| `int` | 整数 |
| `float` | 浮点数 |
| `date` | ISO 日期格式 |

## 二十一、消息分页加载功能详情

### 21.1 API 端点

```
GET /messages/{session_id}/load?message_id=xxx&limit=20
```

### 21.2 加载模式

| 场景 | 参数 | 说明 |
|------|------|------|
| 首次加载 | 无参数 | 获取最新 20 条 |
| 向上滚动 | `message_id=xxx` | 从该消息之前的消息开始 |
| 自定义数量 | `limit=50` | 获取 50 条 |

## 二十二、FAQ 导出功能详情

### 22.1 API 端点

```
GET /faq/export?format=csv&status=published
```

### 22.2 支持的格式

| 格式 | MIME 类型 |
|------|-----------|
| CSV | `text/csv; charset=utf-8` |
| JSON | `application/json` |
| Excel | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |

## 二十三、继续接收流功能详情

### 23.1 API 端点

```
GET /sessions/{session_id}/stream-info
GET /sessions/{session_id}/continue-stream
```

### 23.2 使用场景

| 场景 | 说明 |
|------|------|
| **断线重连** | 客户端网络断开后，使用 `since` 参数继续接收 |
| **多端同步** | 多个设备同时查看同一会话 |
| **事件回放** | 使用 `since` 参数获取历史事件 |

### 23.3 实现方式

- Redis 存储流事件（24小时 TTL）
- SSE 流式返回
- 支持 `since` 参数从指定位置开始
- 超时控制避免永久挂起
