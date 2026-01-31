"""API v1 路由"""

from fastapi import APIRouter

from app.api.v1 import (
    agents,
    api_keys,
    auth,
    chat,
    # chunks,  # 暂时禁用（未完成）
    # documents,  # 暂时禁用（未完成）
    # elasticsearch,  # 暂时禁用（未完成）
    # faq,  # 暂时禁用（未完成）
    # initialization,  # 暂时禁用（未完成）
    knowledge,
    # knowledge_faq,  # 暂时禁用（未完成）
    # knowledge_tags,  # 暂时禁用（未完成）
    mcp_services,
    messages,
    # models,  # 暂时禁用（未完成）
    sessions,
    # system,  # 暂时禁用（未完成）
    # tenant_config,  # 暂时禁用（未完成）
    tenants,
    # vectors,  # 暂时禁用（未完成）
)

router = APIRouter()

# 注册子路由（仅注册已完成的）
router.include_router(chat.router)
router.include_router(auth.router)
router.include_router(api_keys.router)
router.include_router(agents.router)
router.include_router(sessions.router)
router.include_router(messages.router)
router.include_router(tenants.router)
router.include_router(mcp_services.router)
router.include_router(knowledge.router)
# 暂时禁用的路由：
# router.include_router(knowledge_faq.router)
# router.include_router(knowledge_tags.router)
# router.include_router(chunks.router)
# router.include_router(models.router)
# router.include_router(faq.router)
# router.include_router(documents.router)
# router.include_router(initialization.router)
# router.include_router(system.router)
# router.include_router(tenant_config.router)
# router.include_router(vectors.router)
# router.include_router(elasticsearch.router)

__all__ = ["router"]
