"""会话服务

提供会话验证和用户 ID 校验功能。
"""

from fastapi import HTTPException

from app.infra.database import session_repository, session_scope


async def validate_session_access(
    session_id: str,
    user_id: str | None,
    tenant_id: int | None,
) -> None:
    """验证会话是否存在，并可选校验用户/租户归属

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID

    Raises:
        HTTPException: 如果会话不存在或无权访问
    """
    async with session_scope() as session:
        repo = session_repository(session)
        session_obj = await repo.get(session_id)
        if session_obj is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id is not None:
            try:
                user_id_int = int(user_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid user_id") from exc

            if session_obj.user_id != user_id_int:
                raise HTTPException(status_code=403, detail="Session access denied")

        if tenant_id is not None and session_obj.tenant_id is not None:
            if session_obj.tenant_id != tenant_id:
                raise HTTPException(status_code=403, detail="Session tenant mismatch")


def resolve_effective_user_id(
    request_user_id: str | None,
    state_user_id: str | None,
) -> str | None:
    """解析有效的用户 ID

    优先使用 request 中的 user_id，如果没有则使用 state 中的 user_id。
    如果两者都存在但不匹配，则抛出异常。

    Args:
        request_user_id: 请求中的用户 ID
        state_user_id: 请求 state 中的用户 ID

    Returns:
        有效的用户 ID

    Raises:
        HTTPException: 如果用户 ID 不匹配
    """
    if (
        state_user_id is not None
        and request_user_id is not None
        and str(state_user_id) != str(request_user_id)
    ):
        raise HTTPException(status_code=403, detail="User mismatch")
    if state_user_id is not None and request_user_id is None:
        return str(state_user_id)
    return request_user_id
