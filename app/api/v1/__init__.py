"""API v1 路由"""

from fastapi import APIRouter

from app.api.v1 import chat, auth

router = APIRouter()

# 注册子路由
router.include_router(chat.router)
router.include_router(auth.router)

__all__ = ["router"]
