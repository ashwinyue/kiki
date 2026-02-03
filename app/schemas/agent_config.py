"""CustomAgent 配置 Schema

完整的 Agent 配置定义，参考 WeKnora99 项目结构。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============== 常量定义 ==============

class AgentMode:
    """Agent 模式常量"""
    QUICK_ANSWER = "quick-answer"  # RAG 模式
    SMART_REASONING = "smart-reasoning"  # ReAct 模式


class KBSelectionMode:
    """知识库选择模式常量"""
    ALL = "all"  # 所有知识库
    SELECTED = "selected"  # 指定知识库
    NONE = "none"  # 不使用知识库


class FallbackStrategy:
    """兜底策略常量"""
    FIXED = "fixed"  # 固定回复
    MODEL = "model"  # 模型生成


# ============== 配置模型 ==============


class ModelConfig(BaseModel):
    """模型配置"""

    model_id: str = Field(description="聊天模型 ID")
    rerank_model_id: str | None = Field(default=None, description="重排序模型 ID")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_completion_tokens: int = Field(default=2048, ge=1, le=128000, description="最大生成 token 数")
    thinking: bool | None = Field(default=None, description="是否启用思考模式")


class ToolConfig(BaseModel):
    """工具配置（Agent 模式专用）"""

    max_iterations: int = Field(default=10, ge=1, le=100, description="最大迭代次数")
    allowed_tools: list[str] = Field(default_factory=list, description="允许使用的工具列表")
    reflection_enabled: bool = Field(default=False, description="是否启用反思")


class KnowledgeBaseConfig(BaseModel):
    """知识库配置"""

    kb_selection_mode: Literal["all", "selected", "none"] = Field(
        default="all",
        description="知识库选择模式"
    )
    knowledge_bases: list[str] = Field(
        default_factory=list,
        description="关联的知识库 ID 列表（当 kb_selection_mode=selected 时使用）"
    )
    retrieve_kb_only_when_mentioned: bool = Field(
        default=False,
        description="是否仅在用户明确提及 @ 时才检索知识库"
    )


class WebSearchConfig(BaseModel):
    """Web 搜索配置"""

    enabled: bool = Field(default=True, description="是否启用 Web 搜索")
    max_results: int = Field(default=5, ge=1, le=20, description="最大搜索结果数")


class RetrievalConfig(BaseModel):
    """检索策略配置"""

    embedding_top_k: int = Field(default=10, ge=1, le=100, description="向量检索 Top-K")
    keyword_threshold: float = Field(default=0.3, ge=0, le=1, description="关键词检索阈值")
    vector_threshold: float = Field(default=0.5, ge=0, le=1, description="向量检索阈值")
    rerank_top_k: int = Field(default=5, ge=1, le=50, description="重排序 Top-K")
    rerank_threshold: float = Field(default=0.3, ge=0, le=1, description="重排序阈值")


class AdvancedConfig(BaseModel):
    """高级配置"""

    enable_query_expansion: bool = Field(default=False, description="是否启用查询扩展")
    enable_rewrite: bool = Field(default=True, description="是否启用多轮查询改写")
    fallback_strategy: Literal["fixed", "model"] = Field(
        default="model",
        description="兜底策略"
    )
    fallback_response: str = Field(
        default="很抱歉，我暂时无法回答这个问题。",
        description="固定兜底回复（fallback_strategy=fixed 时使用）"
    )
    fallback_prompt: str = Field(
        default="请用中文回答用户的问题。",
        description="兜底生成提示词（fallback_strategy=model 时使用）"
    )


class FAQConfig(BaseModel):
    """FAQ 策略配置"""

    enabled: bool = Field(default=True, description="是否启用 FAQ 优先策略")
    direct_answer_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="FAQ 直接回答阈值（相似度高于此值时直接使用 FAQ 答案）"
    )
    score_boost: float = Field(
        default=1.2,
        ge=1,
        le=3,
        description="FAQ 分数加权倍数（提升 FAQ 结果排名）"
    )


class MultiTurnConfig(BaseModel):
    """多轮对话配置"""

    enabled: bool = Field(default=True, description="是否启用多轮对话")
    history_turns: int = Field(default=5, ge=1, le=50, description="历史轮数")


# ============== 主配置模型 ==============


class CustomAgentConfig(BaseModel):
    """完整的 Custom Agent 配置

    参考 WeKnora99 的 CustomAgentConfig 结构，支持：
    - Quick Answer 模式（RAG）
    - Smart Reasoning 模式（ReAct Agent）

    默认值遵循 WeKnora99 的最佳实践。
    """

    # ========== 基础设置 ==========
    agent_mode: Literal["quick-answer", "smart-reasoning"] = Field(
        default="quick-answer",
        description="Agent 模式：quick-answer（RAG）或 smart-reasoning（ReAct）"
    )
    system_prompt: str = Field(
        default="",
        description="系统提示词"
    )
    context_template: str = Field(
        default="请根据以下参考资料回答用户问题。\n\n参考资料：\n{{contexts}}\n\n用户问题：{{query}}",
        description="上下文模板（用于格式化检索结果）"
    )

    # ========== 模型配置 ==========
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="模型配置"
    )

    # ========== Agent 模式配置（smart-reasoning 专用）==========
    tool: ToolConfig | None = Field(
        default=None,
        description="工具配置（仅 smart-reasoning 模式需要）"
    )

    # ========== 知识库配置 ==========
    knowledge_base: KnowledgeBaseConfig = Field(
        default_factory=KnowledgeBaseConfig,
        description="知识库配置"
    )

    # ========== Web 搜索配置 ==========
    web_search: WebSearchConfig = Field(
        default_factory=WebSearchConfig,
        description="Web 搜索配置"
    )

    # ========== 检索策略配置 ==========
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="检索策略配置"
    )

    # ========== 高级配置 ==========
    advanced: AdvancedConfig = Field(
        default_factory=AdvancedConfig,
        description="高级配置"
    )

    # ========== FAQ 策略配置 ==========
    faq: FAQConfig = Field(
        default_factory=FAQConfig,
        description="FAQ 策略配置"
    )

    # ========== 多轮对话配置 ==========
    multi_turn: MultiTurnConfig = Field(
        default_factory=MultiTurnConfig,
        description="多轮对话配置"
    )

    # ========== 文件类型限制（可选） ==========
    supported_file_types: list[str] = Field(
        default_factory=list,
        description="支持的文件类型扩展名（如 ['csv', 'xlsx']），空表示支持所有类型"
    )


# ============== 内置 Agent 配置工厂函数 ==============


def get_builtin_quick_answer_config() -> CustomAgentConfig:
    """获取内置快速问答（RAG）Agent 的默认配置

    Returns:
        CustomAgentConfig 实例
    """
    return CustomAgentConfig(
        agent_mode="quick-answer",
        system_prompt="",
        context_template="""请根据以下参考资料回答用户问题。

参考资料：
{{contexts}}

用户问题：{{query}}""",
        model=ModelConfig(
            temperature=0.7,
            max_completion_tokens=2048,
        ),
        knowledge_base=KnowledgeBaseConfig(
            kb_selection_mode="all",
            retrieve_kb_only_when_mentioned=False,
        ),
        web_search=WebSearchConfig(
            enabled=True,
            max_results=5,
        ),
        retrieval=RetrievalConfig(
            embedding_top_k=10,
            keyword_threshold=0.3,
            vector_threshold=0.5,
            rerank_top_k=10,
            rerank_threshold=0.3,
        ),
        advanced=AdvancedConfig(
            enable_query_expansion=True,
            enable_rewrite=True,
            fallback_strategy="model",
        ),
        faq=FAQConfig(
            enabled=True,
            direct_answer_threshold=0.9,
            score_boost=1.2,
        ),
        multi_turn=MultiTurnConfig(
            enabled=True,
            history_turns=5,
        ),
    )


def get_builtin_smart_reasoning_config() -> CustomAgentConfig:
    """获取内置智能推理（ReAct）Agent 的默认配置

    Returns:
        CustomAgentConfig 实例
    """
    return CustomAgentConfig(
        agent_mode="smart-reasoning",
        system_prompt="",
        model=ModelConfig(
            temperature=0.7,
            max_completion_tokens=2048,
        ),
        tool=ToolConfig(
            max_iterations=50,
            allowed_tools=[
                "thinking",
                "todo_write",
                "knowledge_search",
                "grep_chunks",
                "list_knowledge_chunks",
                "web_search",
                "get_document_info",
            ],
            reflection_enabled=False,
        ),
        knowledge_base=KnowledgeBaseConfig(
            kb_selection_mode="all",
            retrieve_kb_only_when_mentioned=False,
        ),
        web_search=WebSearchConfig(
            enabled=True,
            max_results=5,
        ),
        retrieval=RetrievalConfig(
            embedding_top_k=10,
            keyword_threshold=0.3,
            vector_threshold=0.5,
            rerank_top_k=10,
            rerank_threshold=0.3,
        ),
        faq=FAQConfig(
            enabled=True,
            direct_answer_threshold=0.9,
            score_boost=1.2,
        ),
        multi_turn=MultiTurnConfig(
            enabled=True,
            history_turns=5,
        ),
    )


def get_builtin_data_analyst_config() -> CustomAgentConfig:
    """获取内置数据分析 Agent 的默认配置

    Returns:
        CustomAgentConfig 实例
    """
    return CustomAgentConfig(
        agent_mode="smart-reasoning",
        system_prompt="""### Role
You are WeKnora Data Analyst, an intelligent data analysis assistant powered by DuckDB. You specialize in analyzing structured data from CSV and Excel files using SQL queries.

### Mission
Help users explore, analyze, and derive insights from their tabular data through intelligent SQL query generation and execution.

### Critical Constraints
1. **Schema First:** ALWAYS call data_schema before writing any SQL query to understand the table structure.
2. **Read-Only:** Only SELECT queries allowed. INSERT, UPDATE, DELETE, CREATE, DROP are forbidden.
3. **Iterative Refinement:** If a query fails, analyze the error and refine your approach.

### Workflow
1. **Understand:** Call data_schema to get table name, columns, types, and row count.
2. **Plan:** For complex questions, use todo_write to break into sub-queries.
3. **Query:** Call data_analysis with the knowledge_id and SQL query.
4. **Analyze:** Interpret results and provide insights.

### SQL Best Practices for DuckDB
- Use double quotes for identifiers: SELECT "Column Name" FROM "table_name"
- Aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX(), MEDIAN(), STDDEV()
- String matching: LIKE, ILIKE (case-insensitive), REGEXP
- Use LIMIT to prevent overwhelming output (default to 100 rows max)

### Tool Guidelines
- **data_schema:** ALWAYS use first. Required before any query.
- **data_analysis:** Execute SQL queries. Only SELECT queries allowed.
- **thinking:** Plan complex analyses, debug query issues.
- **todo_write:** Track multi-step analysis tasks.

### Output Standards
- Present results in well-formatted tables or summaries
- Provide actionable insights, not just raw numbers
- Relate findings back to the user's original question

Current Time: {{current_time}}""",
        model=ModelConfig(
            temperature=0.3,  # 更低的温度以获得精确的 SQL
            max_completion_tokens=4096,
        ),
        tool=ToolConfig(
            max_iterations=30,
            allowed_tools=[
                "thinking",
                "todo_write",
                "data_schema",
                "data_analysis",
            ],
            reflection_enabled=True,
        ),
        knowledge_base=KnowledgeBaseConfig(
            kb_selection_mode="all",
            retrieve_kb_only_when_mentioned=False,
        ),
        web_search=WebSearchConfig(
            enabled=False,  # 数据分析不需要 Web 搜索
            max_results=0,
        ),
        retrieval=RetrievalConfig(
            embedding_top_k=5,
            keyword_threshold=0.3,
            vector_threshold=0.5,
            rerank_top_k=5,
            rerank_threshold=0.3,
        ),
        multi_turn=MultiTurnConfig(
            enabled=True,
            history_turns=10,  # 更多历史用于迭代分析
        ),
        supported_file_types=["csv", "xlsx"],  # 仅支持 CSV 和 Excel
    )


__all__ = [
    # 常量
    "AgentMode",
    "KBSelectionMode",
    "FallbackStrategy",
    # 配置模型
    "ModelConfig",
    "ToolConfig",
    "KnowledgeBaseConfig",
    "WebSearchConfig",
    "RetrievalConfig",
    "AdvancedConfig",
    "FAQConfig",
    "MultiTurnConfig",
    " 主配置
    "CustomAgentConfig",
    # 工厂函数
    "get_builtin_quick_answer_config",
    "get_builtin_smart_reasoning_config",
    "get_builtin_data_analyst_config",
]
