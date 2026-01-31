# Kiki 知识库与模型管理 API 实现计划

## 概述

为 Kiki 项目添加知识库管理 API 和模型管理 API，完全对齐 WeKnora99 的接口设计。

## 目标

1. **知识库管理 API** - 提供知识库的 CRUD 操作、混合搜索
2. **知识条目管理 API** - 支持从文件/URL 创建知识、查询列表
3. **模型管理 API** - 多类型模型配置（Embedding、Rerank、KnowledgeQA）
4. **利用现有能力** - 复用 `app/agent/capabilities/rag.py` 的 PgVectorStore 和 LangChain 集成

## WeKnora99 API 协议（完全对齐）

### 知识库管理

```python
# POST /knowledge-bases - 创建知识库
{
    "name": str,
    "description": str,
    "chunking_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["."],
        "enable_multimodal": true
    },
    "embedding_model_id": str,
    "summary_model_id": str
}

# GET /knowledge-bases - 查询知识库列表

# GET /knowledge-bases/{kb_id} - 获取知识库详情

# PUT /knowledge-bases/{kb_id} - 更新知识库

# DELETE /knowledge-bases/{kb_id} - 删除知识库

# GET /knowledge-bases/{kb_id}/hybrid-search - 混合搜索
{
    "query_text": str,
    "vector_threshold": 0.5,
    "keyword_threshold": 0.3,
    "match_count": 5
}
```

### 知识条目管理

```python
# POST /knowledge-bases/{kb_id}/knowledge/file - 从文件创建
Content-Type: multipart/form-data
{ "file": file, "enable_multimodel": "true" }

# POST /knowledge-bases/{kb_id}/knowledge/url - 从URL创建
{
    "url": str,
    "enable_multimodal": bool
}

# GET /knowledge-bases/{kb_id}/knowledge - 知识条目列表
Query: ?page=1&page_size=20

# GET /knowledge/{knowledge_id} - 知识条目详情

# DELETE /knowledge/{knowledge_id} - 删除知识条目
```

### 模型管理

```python
# POST /models - 创建模型
{
    "name": str,
    "type": str,  # KnowledgeQA, Embedding, Rerank
    "source": str,  # local, openai, aliyun, zhipu
    "description": str,
    "parameters": {
        "base_url": str,
        "api_key": str
    },
    "is_default": bool
}

# GET /models - 模型列表

# GET /models/{model_id} - 模型详情
```

## 架构设计

### 分层架构（遵循项目现有模式）

```
┌─────────────────────────────────────────────────────────────┐
│                      API 路由层 (FastAPI)                    │
│  app/api/v1/knowledge.py  |  app/api/v1/models.py           │
├─────────────────────────────────────────────────────────────┤
│                     业务逻辑层 (Service)                      │
│  app/services/knowledge_service.py  |  app/services/...     │
│     利用: app/agent/capabilities/rag.py (PgVectorStore)      │
│     利用: langchain 文档加载器 + 分割器                      │
├─────────────────────────────────────────────────────────────┤
│                     数据访问层 (Repository)                   │
│  app/repositories/knowledge.py  |  app/repositories/model.py│
├─────────────────────────────────────────────────────────────┤
│                     数据模型层 (Model/Schema)                │
│  app/models/knowledge.py (已有) | app/schemas/knowledge.py   │
└─────────────────────────────────────────────────────────────┘
```

### 复用现有能力

| 能力 | 位置 | 说明 |
|------|------|------|
| PgVectorStore | `app/agent/capabilities/rag.py` | PostgreSQL + pgvector 向量存储 |
| Embeddings | `app/llm/embeddings.py` | 多提供商嵌入服务 |
| LangChain Loaders | `langchain_community.document_loaders` | 文档加载 |
| LangChain Splitters | `langchain_text_splitters` | 文档分块 |

## 实现方案

### 第一阶段：Schema 定义

**文件：`app/schemas/knowledge.py`**

```python
# 知识库相关 (完全对齐 WeKnora99)
class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = ["."]
    enable_multimodal: bool = True

class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str
    chunking_config: ChunkingConfig | None = None
    embedding_model_id: str
    summary_model_id: str

class KnowledgeBaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    chunking_config: ChunkingConfig | None = None
    embedding_model_id: str | None = None
    summary_model_id: str | None = None

class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str | None
    chunking_config: dict
    embedding_model_id: str
    summary_model_id: str
    knowledge_count: int = 0
    created_at: datetime
    updated_at: datetime

# 知识条目相关
class KnowledgeCreateFromFile(BaseModel):
    enable_multimodel: bool = True

class KnowledgeCreateFromURL(BaseModel):
    url: str
    enable_multimodel: bool = True

class KnowledgeResponse(BaseModel):
    id: str
    knowledge_base_id: str
    type: str
    title: str
    source: str
    parse_status: str
    file_name: str | None
    file_size: int | None
    chunk_count: int = 0
    created_at: datetime

# 搜索相关
class HybridSearchRequest(BaseModel):
    query_text: str
    vector_threshold: float = 0.5
    keyword_threshold: float = 0.3
    match_count: int = 5

class HybridSearchResult(BaseModel):
    content: str
    score: float
    chunk_id: str
    knowledge_id: str
    metadata: dict
```

**文件：`app/schemas/model.py`**

```python
# 模型参数 (对齐 WeKnora99)
class ModelParameters(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    dimensions: int | None = None

class ModelCreate(BaseModel):
    name: str
    type: str  # Embedding, Rerank, KnowledgeQA, Chat, VLLM
    source: str = "local"  # local, openai, aliyun, zhipu, remote
    description: str
    parameters: ModelParameters = ModelParameters()
    is_default: bool = False

class ModelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    parameters: ModelParameters | None = None
    is_default: bool | None = None

class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    source: str
    description: str | None
    parameters: dict
    is_default: bool
    is_builtin: bool
    status: str
    created_at: datetime
```

### 第二阶段：Repository 层

**文件：`app/repositories/knowledge.py`**

```python
class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    - get_by_tenant(id, tenant_id)
    - list_paginated_by_tenant(tenant_id, params)
    - create_with_user(data, user_id)
    - soft_delete(id)

class KnowledgeRepository(BaseRepository[Knowledge]):
    - get_by_kb(knowledge_id, kb_id)
    - list_by_kb(kb_id, params)
    - create_from_file(kb_id, file, ...)
    - create_from_url(kb_id, url, ...)
    - soft_delete(id)

class ChunkRepository(BaseRepository[Chunk]):
    - hybrid_search(kb_id, query, thresholds)
    - list_by_knowledge(knowledge_id)
```

**文件：`app/repositories/model.py`**

```python
class ModelRepository(BaseRepository[Model]):
    - get_by_tenant(id, tenant_id)
    - list_by_type(model_type, tenant_id)
    - get_default(model_type, tenant_id)
    - set_default(id)
    - list_paginated_by_tenant(tenant_id, params)
```

### 第三阶段：Service 层

**文件：`app/services/knowledge_service.py`**

```python
class KnowledgeBaseService:
    - create_knowledge_base(data, tenant_id)
    - get_knowledge_base(kb_id, tenant_id)
    - list_knowledge_bases(tenant_id, params)
    - update_knowledge_base(kb_id, data, tenant_id)
    - delete_knowledge_base(kb_id, tenant_id)
    - hybrid_search(kb_id, query, thresholds)

class KnowledgeService:
    - create_from_file(kb_id, file, tenant_id)
    - create_from_url(kb_id, url, tenant_id)
    - get_knowledge(knowledge_id, tenant_id)
    - list_knowledge(kb_id, tenant_id, params)
    - delete_knowledge(knowledge_id, tenant_id)
```

**文件：`app/services/model_service.py`**

```python
class ModelService:
    - create_model(data, tenant_id)
    - get_model(model_id, tenant_id)
    - list_models(tenant_id, model_type)
    - update_model(model_id, data, tenant_id)
    - delete_model(model_id, tenant_id)
    - set_default(model_id, tenant_id)
```

### 第四阶段：API 路由层

**文件：`app/api/v1/knowledge.py`**

```python
# 知识库管理 (完全对齐 WeKnora99)
POST   /knowledge-bases           # 创建知识库
GET    /knowledge-bases           # 知识库列表
GET    /knowledge-bases/{kb_id}   # 知识库详情
PUT    /knowledge-bases/{kb_id}   # 更新知识库
DELETE /knowledge-bases/{kb_id}   # 删除知识库
GET    /knowledge-bases/{kb_id}/hybrid-search  # 混合搜索

# 知识条目管理
POST   /knowledge-bases/{kb_id}/knowledge/file  # 从文件创建
POST   /knowledge-bases/{kb_id}/knowledge/url   # 从URL创建
GET    /knowledge-bases/{kb_id}/knowledge       # 知识条目列表
GET    /knowledge/{knowledge_id}    # 知识条目详情
DELETE /knowledge/{knowledge_id}    # 删除知识条目
```

**文件：`app/api/v1/models.py`**

```python
POST   /models              # 创建模型
GET    /models              # 模型列表
GET    /models/{model_id}   # 模型详情
PATCH  /models/{model_id}   # 更新模型
DELETE /models/{model_id}   # 删除模型
```

## 文件清单

### 新建文件

| 文件路径 | 说明 | 关键依赖 |
|---------|------|----------|
| `app/schemas/knowledge.py` | 知识库相关 Schema | pydantic |
| `app/schemas/model.py` | 模型相关 Schema | pydantic |
| `app/repositories/knowledge.py` | 知识库 Repository | BaseRepository |
| `app/repositories/model.py` | 模型 Repository | BaseRepository |
| `app/services/knowledge_service.py` | 知识库服务 | PgVectorStore, LangChain |
| `app/services/model_service.py` | 模型服务 | - |
| `app/api/v1/knowledge.py` | 知识库 API 路由 | FastAPI |
| `app/api/v1/models.py` | 模型 API 路由 | FastAPI |

### 修改文件

| 文件路径 | 修改内容 |
|---------|----------|
| `app/schemas/__init__.py` | 添加知识库和模型相关导出 |
| `app/api/v1/__init__.py` | 注册新路由 |
| `app/agent/capabilities/rag.py` | (可选) 增强：添加混合搜索方法 |

## 关键设计决策

### 1. 模型类型枚举（完全对齐 WeKnora99）

```python
class ModelType(str, Enum):
    EMBEDDING = "Embedding"       # 向量嵌入
    RERANK = "Rerank"             # 重排序
    KNOWLEDGE_QA = "KnowledgeQA"  # 知识问答
    CHAT = "Chat"                 # 对话
    VLLM = "VLLM"                 # 大语言模型
```

### 2. 模型来源枚举（完全对齐 WeKnora99）

```python
class ModelSource(str, Enum):
    LOCAL = "local"      # 本地部署
    OPENAI = "openai"    # OpenAI
    ALIYUN = "aliyun"    # 阿里云
    ZHIPU = "zhipu"      # 智谱AI
    REMOTE = "remote"    # 远程服务
```

### 3. 文档处理（使用 LangChain）

```python
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF
    Docx2txtLoader,        # DOCX
    UnstructuredMarkdownLoader,  # Markdown
    TextLoader,            # TXT
    UnstructuredURLLoader, # URL
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文档加载器映射
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}

# 文档分块
def split_documents(documents, config):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )
    return splitter.split_documents(documents)
```

### 4. 向量存储（复用现有 PgVectorStore）

```python
from app.agent.capabilities.rag import PgVectorStore, VectorStoreConfig

# 为每个知识库创建独立的向量存储
async def get_vector_store(kb_id, embedding_model_id):
    config = VectorStoreConfig(
        collection_name=f"kb_{kb_id}",
        embedding_model=embedding_model_id,
    )
    return PgVectorStore(config=config)
```

### 5. 混合搜索（向量 + 关键词）

```python
async def hybrid_search(kb_id, query, thresholds):
    # 1. 向量搜索
    vector_results = await vector_store.similarity_search(
        query, k=match_count, score_threshold=vector_threshold
    )
    # 2. 关键词搜索 (PostgreSQL full-text search)
    keyword_results = await keyword_search(query, kb_id, keyword_threshold)
    # 3. 结果合并与重排序
    return merge_and_rerank(vector_results, keyword_results)
```

### 6. 软删除策略

所有删除操作采用软删除，设置 `deleted_at` 字段而非物理删除。

### 7. 租户隔离

所有查询自动添加 `tenant_id` 过滤，确保多租户数据隔离。

## 验证计划

### 依赖安装

```bash
# PostgreSQL + pgvector 支持
uv add -E pgvector langchain-postgres pgvector

# LangChain 文档加载器
uv add -E loaders langchain-community langchain-text-splitters

# 文档处理
uv add python-magic unstructured[all-in-one]
```

### API 测试步骤

```bash
# 1. 启动服务
uv run uvicorn app.main:app --reload

# 2. 创建嵌入模型
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "text-embedding-3-small",
    "type": "Embedding",
    "source": "openai",
    "description": "OpenAI 嵌入模型",
    "parameters": {"api_key": "sk-xxx"},
    "is_default": true
  }'

# 3. 创建知识库
curl -X POST http://localhost:8000/api/v1/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "测试知识库",
    "description": "API 测试知识库",
    "embedding_model_id": "<model_id>",
    "summary_model_id": "<model_id>"
  }'

# 4. 上传文档
curl -X POST http://localhost:8000/api/v1/knowledge-bases/<kb_id>/knowledge/file \
  -F "file=@test.pdf" \
  -F "enable_multimodel=true"

# 5. 混合搜索
curl "http://localhost:8000/api/v1/knowledge-bases/<kb_id>/hybrid-search?query_text=测试&match_count=5"

# 6. 获取知识列表
curl http://localhost:8000/api/v1/knowledge-bases/<kb_id>/knowledge
```

## 依赖关系

```
knowledge.py (API)
    ↓ 依赖
knowledge_service.py (Service)
    ↓ 依赖
knowledge.py (Repository)
    ↓ 依赖
knowledge.py (Model)
```

## 实施顺序

1. **Schema 定义** - 首先定义数据结构
2. **Repository 层** - 实现数据访问
3. **Service 层** - 实现业务逻辑
4. **API 层** - 暴露 HTTP 接口
5. **路由注册** - 将新路由集成到主应用
6. **测试验证** - 确保功能正常
