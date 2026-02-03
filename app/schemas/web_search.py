"""网络搜索相关模式

提供网络搜索 API 的请求/响应模型，对齐 WeKnora 的 web_search.go
"""

from datetime import datetime

from pydantic import BaseModel, Field


class WebSearchConfig(BaseModel):
    """Web 搜索配置

    对应 WeKnora 的 types.WebSearchConfig
    """

    provider: str = Field("duckduckgo", description="搜索引擎提供商ID")
    api_key: str | None = Field(None, description="API密钥（如果需要）")
    max_results: int = Field(5, ge=1, le=20, description="最大搜索结果数")
    include_date: bool = Field(False, description="是否包含日期")
    compression_method: str = Field(
        "none",
        description="压缩方法：none, summary, extract, rag",
    )
    blacklist: list[str] = Field(default_factory=list, description="黑名单规则列表")
    # RAG压缩相关配置
    embedding_model_id: str | None = Field(None, description="嵌入模型ID（用于RAG压缩）")
    embedding_dimension: int = Field(0, description="嵌入维度（用于RAG压缩）")
    rerank_model_id: str | None = Field(None, description="重排模型ID（用于RAG压缩）")
    document_fragments: int = Field(3, ge=1, description="文档片段数量（用于RAG压缩）")


class WebSearchResult(BaseModel):
    """单个 Web 搜索结果

    对应 WeKnora 的 types.WebSearchResult
    """

    title: str = Field(..., description="搜索结果标题")
    url: str = Field(..., description="结果URL")
    snippet: str = Field(..., description="摘要片段")
    content: str | None = Field(None, description="完整内容（可选，需要额外抓取）")
    source: str = Field(..., description="来源（如：duckduckgo等）")
    published_at: datetime | None = Field(None, description="发布时间（如果有）")


class WebSearchProviderInfo(BaseModel):
    """Web 搜索提供商信息

    对应 WeKnora 的 types.WebSearchProviderInfo
    """

    id: str = Field(..., description="提供商ID")
    name: str = Field(..., description="提供商名称")
    free: bool = Field(..., description="是否免费")
    requires_api_key: bool = Field(..., description="是否需要API密钥")
    description: str = Field(..., description="描述")
    api_url: str | None = Field(None, description="API地址（可选）")
    available: bool = Field(True, description="当前是否可用")


class WebSearchRequest(BaseModel):
    """Web 搜索请求"""

    query: str = Field(..., min_length=1, description="搜索查询")
    provider: str | None = Field(None, description="搜索引擎（不指定则使用默认）")
    max_results: int | None = Field(None, ge=1, le=20, description="最大结果数")
    include_date: bool = Field(False, description="是否包含日期")
    blacklist: list[str] = Field(default_factory=list, description="黑名单规则")


class WebSearchResponse(BaseModel):
    """Web 搜索响应"""

    results: list[WebSearchResult] = Field(default_factory=list, description="搜索结果")
    provider: str = Field(..., description="使用的搜索引擎")
    query: str = Field(..., description="搜索查询")
    total: int = Field(..., description="结果总数")


class WebSearchProvidersResponse(BaseModel):
    """Web 搜索提供商列表响应"""

    providers: list[WebSearchProviderInfo] = Field(
        default_factory=list, description="提供商列表"
    )
    default_provider: str = Field(..., description="默认提供商")


class WebSearchCompressRequest(BaseModel):
    """RAG 压缩请求（预留接口）"""

    session_id: str = Field(..., description="会话ID")
    temp_kb_id: str = Field("", description="临时知识库ID（空字符串则创建新的）")
    questions: list[str] = Field(..., min_length=1, description="问题列表")
    results: list[WebSearchResult] = Field(..., min_length=1, description="搜索结果")
    config: WebSearchConfig = Field(..., description="搜索配置")


class WebSearchCompressResponse(BaseModel):
    """RAG 压缩响应"""

    results: list[WebSearchResult] = Field(default_factory=list, description="压缩后的结果")
    temp_kb_id: str = Field(..., description="临时知识库ID")
    knowledge_ids: list[str] = Field(default_factory=list, description="知识ID列表")
