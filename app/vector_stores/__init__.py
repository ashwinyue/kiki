"""向量存储模块

提供统一的向量存储接口，支持多种向量数据库：
- Qdrant: 高性能向量数据库，支持本地和云端部署
- Pinecone: 托管向量搜索服务
- Elasticsearch: 企业级搜索引擎，支持向量和全文搜索
- 内存模式: 开发/测试环境使用

使用示例:
```python
from app.vector_stores import create_vector_store, VectorStoreConfig

# 创建向量存储
config = VectorStoreConfig(collection_name="documents")
store = create_vector_store("qdrant", config)

# 添加文档
await store.add_documents(documents)

# 搜索
results = await store.similarity_search("查询内容", k=5)
```
"""

from app.vector_stores.base import BaseVectorStore, VectorStoreConfig
from app.vector_stores.factory import VectorStoreFactory, create_vector_store

__all__ = [
    "BaseVectorStore",
    "VectorStoreConfig",
    "VectorStoreFactory",
    "create_vector_store",
]
