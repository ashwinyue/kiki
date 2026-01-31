"""API v1 路由"""

from fastapi import APIRouter

from app.api.v1 import agents, api_keys, auth, chat, evaluation, mcp_services, tenants, tools

router = APIRouter()

# 注册子路由
router.include_router(chat.router)
router.include_router(auth.router)
router.include_router(api_keys.router)
router.include_router(tools.router)
router.include_router(agents.router)
router.include_router(evaluation.router)
router.include_router(tenants.router)
router.include_router(mcp_services.router)

__all__ = ["router"]
