"""LLM 服务模块

提供 LLM 模型注册表、使用 LangChain 内置重试机制、结构化输出支持。
集成多模型成本优化路由系统、Embedding 服务、成本追踪。
"""

from app.llm.cost_tracker import (  # type: ignore
    CostRecord,
    CostSummary,
    CostTracker,
    check_budget,
    get_cost_summary,
    get_cost_tracker,
    get_model_pricing,
    record_llm_usage,
    register_model_pricing,
    set_cost_tracker,
    track_llm_call,
)
from app.llm.embeddings import (  # type: ignore
    DashScopeEmbeddings,
    EmbeddingProvider,
    get_embeddings,
)
from app.llm.providers import (
    LLMPriority,
    LLMProviderError,
    get_llm_for_task,
)
from app.llm.registry import (
    LLMRegistry,
    _init_default_models,
)
from app.llm.service import (
    LLMService,
    get_llm_service,
    resolve_provider,
)

__all__ = [
    # registry
    "LLMRegistry",
    "_init_default_models",
    # service
    "LLMService",
    "get_llm_service",
    "resolve_provider",
    # providers
    "LLMPriority",
    "LLMProviderError",
    "get_llm_for_task",
    # embeddings
    "EmbeddingProvider",
    "DashScopeEmbeddings",
    "get_embeddings",
    # cost tracker
    "CostTracker",
    "CostRecord",
    "CostSummary",
    "get_cost_tracker",
    "set_cost_tracker",
    "get_model_pricing",
    "register_model_pricing",
    "track_llm_call",
    "record_llm_usage",
    "get_cost_summary",
    "check_budget",
]
