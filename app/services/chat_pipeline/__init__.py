"""聊天管道服务

对齐 WeKnora99 聊天管道架构，实现 RAG 搜索流程。

Pipeline 流程:
1. query_rewrite: 查询重写
2. search: 知识库搜索 (混合搜索 + 网络搜索)
3. rerank: 重排序 (MMR 多样性)
4. filter_top_k: 过滤 Top K
5. build_context: 构建上下文
6. chat_completion: LLM 生成

使用示例:
    ```python
    from app.services.chat_pipeline import ChatPipeline, ChatPipelineConfig

    config = ChatPipelineConfig(
        tenant_id=1,
        session_id="xxx",
        knowledge_base_ids=["kb1"],
    )

    pipeline = ChatPipeline(
        config=config,
        llm_service=llm_service,
        knowledge_service=knowledge_service,
    )

    result = await pipeline.run(query="什么是 Python?")
    ```

基于 LangGraph 的新 API:
    ```python
    from app.services.chat_pipeline import ChatPipelineGraph, create_pipeline_graph

    graph = create_pipeline_graph(
        llm_service=llm_service,
        knowledge_service=knowledge_service,
        message_service=message_service,
    )

    result = await graph.run(config, "什么是 Python?")

    # 获取 Mermaid 流程图
    print(graph.get_graph_mermaid())
    ```
"""

from app.services.chat_pipeline.graph import (
    ChatPipelineGraph,
    PipelineState,
    create_pipeline_graph,
)
from app.services.chat_pipeline.pipeline import (
    ChatPipeline,
    ChatPipelineBuilder,
    PipelineError,
    create_pipeline,
)
from app.services.chat_pipeline.stages import (
    ContextBuildStage,
    QueryRewriteStage,
    RerankStage,
    SearchStage,
)
from app.services.chat_pipeline.types import (
    ChatContext,
    ChatPipelineConfig,
    ChatPipelineResult,
    History,
    SearchTarget,
    SearchTargetType,
    SearchResult,
    SearchResultType,
)

__all__ = [
    # Pipeline（原始实现，向后兼容）
    "ChatPipeline",
    "ChatPipelineBuilder",
    "PipelineError",
    "create_pipeline",
    # Pipeline Graph（基于 LangGraph 的新实现）
    "ChatPipelineGraph",
    "PipelineState",
    "create_pipeline_graph",
    # Stages
    "QueryRewriteStage",
    "SearchStage",
    "RerankStage",
    "ContextBuildStage",
    # Types
    "ChatContext",
    "ChatPipelineConfig",
    "ChatPipelineResult",
    "SearchResult",
    "SearchResultType",
    "SearchTarget",
    "SearchTargetType",
    "History",
]
