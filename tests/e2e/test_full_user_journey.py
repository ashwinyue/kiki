"""完整用户旅程 E2E 测试

模拟真实用户的完整使用流程：
1. 用户注册
2. 用户登录
3. 创建聊天会话
4. 发送消息并接收回复
5. 查看聊天历史
6. 删除会话
"""

import pytest

from fastapi.testclient import TestClient


@pytest.mark.e2e
class TestFullUserJourney:
    """完整用户旅程测试"""

    def test_new_user_complete_flow(self, e2e_client: TestClient) -> None:
        """测试新用户完整流程：注册 -> 登录 -> 聊天 -> 查看历史"""
        # ========== Step 1: 用户注册 ==========
        register_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "full_name": "New User",
        }

        response = e2e_client.post("/api/v1/auth/register", json=register_data)
        assert response.status_code == 200
        register_result = response.json()

        assert register_result["email"] == register_data["email"]
        assert register_result["full_name"] == register_data["full_name"]
        assert "access_token" in register_result
        assert register_result["token_type"] == "bearer"

        access_token = register_result["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # ========== Step 2: 验证用户信息 ==========
        response = e2e_client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        user_data = response.json()

        assert user_data["email"] == register_data["email"]
        assert user_data["full_name"] == register_data["full_name"]
        assert user_data["is_active"] is True

        # ========== Step 3: 创建聊天会话 ==========
        session_data = {"name": "我的第一个对话"}
        response = e2e_client.post(
            "/api/v1/auth/sessions",
            data={"name": session_data["name"]},  # Form data
            headers=headers,
        )
        assert response.status_code == 200
        session_result = response.json()

        assert "session_id" in session_result
        assert session_result["name"] == session_data["name"]
        assert "token" in session_result

        session_id = session_result["session_id"]

        # ========== Step 4: 发送聊天消息（同步） ==========
        # 注意：如果 LLM 未配置，可能返回 500
        chat_data = {
            "message": "你好，请介绍一下你自己",
            "session_id": session_id,
        }

        response = e2e_client.post("/api/v1/chat", json=chat_data)
        # LLM 未配置时可能失败，这是预期的
        assert response.status_code in (200, 500)

        if response.status_code == 200:
            chat_result = response.json()
            assert "content" in chat_result
            assert chat_result["session_id"] == session_id
        else:
            # 跳过后续依赖聊天响应的测试
            pytest.skip("LLM 未配置，跳过聊天相关测试")

        # ========== Step 5: 获取聊天历史 ==========
        response = e2e_client.get(f"/api/v1/chat/history/{session_id}")
        assert response.status_code == 200
        history_result = response.json()

        assert "messages" in history_result
        assert history_result["session_id"] == session_id
        # 应该至少有用户消息
        assert len(history_result["messages"]) >= 1

        # ========== Step 6: 获取会话列表 ==========
        response = e2e_client.get("/api/v1/auth/sessions", headers=headers)
        assert response.status_code == 200
        sessions_list = response.json()

        assert isinstance(sessions_list, list)
        assert len(sessions_list) >= 1
        # 验证我们创建的会话在列表中
        assert any(s["session_id"] == session_id for s in sessions_list)

        # ========== Step 7: 删除会话 ==========
        response = e2e_client.delete(f"/api/v1/auth/sessions/{session_id}", headers=headers)
        assert response.status_code == 200
        delete_result = response.json()
        assert delete_result["status"] == "success"

        # ========== Step 8: 验证会话已删除 ==========
        response = e2e_client.get("/api/v1/auth/sessions", headers=headers)
        assert response.status_code == 200
        sessions_list = response.json()
        # 会话不再在列表中
        assert not any(s["session_id"] == session_id for s in sessions_list)

    def test_login_flow(self, e2e_client: TestClient) -> None:
        """测试登录流程"""
        # 先注册一个用户
        user_data = {
            "email": "loginflow@example.com",
            "password": "LoginPass123!",
            "full_name": "Login Flow User",
        }
        e2e_client.post("/api/v1/auth/register", json=user_data)

        # ========== 测试表单登录 ==========
        response = e2e_client.post(
            "/api/v1/auth/login",
            data={
                "username": user_data["email"],
                "password": user_data["password"],
            },
        )
        assert response.status_code == 200
        token_result = response.json()
        assert "access_token" in token_result

        # ========== 测试 JSON 登录 ==========
        response = e2e_client.post(
            "/api/v1/auth/login/json",
            json={
                "username": user_data["email"],
                "password": user_data["password"],
            },
        )
        assert response.status_code == 200
        token_result = response.json()
        assert "access_token" in token_result

        # ========== 测试错误密码 ==========
        response = e2e_client.post(
            "/api/v1/auth/login/json",
            json={
                "username": user_data["email"],
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401

    def test_multiple_conversations(self, e2e_client: TestClient) -> None:
        """测试多会话场景"""
        # 注册并登录
        register_resp = e2e_client.post(
            "/api/v1/auth/register",
            json={
                "email": "multiuser@example.com",
                "password": "MultiPass123!",
                "full_name": "Multi User",
            },
        )
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 创建多个会话
        session_ids = []
        session_names = ["工作对话", "学习对话", "生活对话"]

        for name in session_names:
            response = e2e_client.post(
                "/api/v1/auth/sessions",
                data={"name": name},
                headers=headers,
            )
            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])

        # 获取会话列表
        response = e2e_client.get("/api/v1/auth/sessions", headers=headers)
        assert response.status_code == 200
        sessions = response.json()

        # 验证所有会话都在列表中
        assert len(sessions) >= len(session_names)
        for session_id in session_ids:
            assert any(s["session_id"] == session_id for s in sessions)

        # 逐个删除会话
        for session_id in session_ids:
            response = e2e_client.delete(
                f"/api/v1/auth/sessions/{session_id}",
                headers=headers,
            )
            assert response.status_code == 200

    def test_context_management(self, e2e_client: TestClient) -> None:
        """测试上下文管理"""
        # 注册
        register_resp = e2e_client.post(
            "/api/v1/auth/register",
            json={
                "email": "contextuser@example.com",
                "password": "ContextPass123!",
                "full_name": "Context User",
            },
        )
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 创建会话
        session_resp = e2e_client.post(
            "/api/v1/auth/sessions",
            data={"name": "Context Test"},
            headers=headers,
        )
        session_id = session_resp.json()["session_id"]

        # 获取上下文统计（初始状态）
        response = e2e_client.get(f"/api/v1/chat/context/{session_id}/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["session_id"] == session_id
        assert stats["message_count"] == 0

        # 清除上下文
        response = e2e_client.delete(f"/api/v1/chat/context/{session_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

    def test_unauthorized_access(self, e2e_client: TestClient) -> None:
        """测试未授权访问"""
        # 不带 token 访问需要认证的接口
        protected_endpoints = [
            "/api/v1/auth/me",
            "/api/v1/auth/sessions",
        ]

        for endpoint in protected_endpoints:
            response = e2e_client.get(endpoint)
            # 可能返回 403 (没有 Bearer header) 或 401
            assert response.status_code in (401, 403)

    def test_invalid_token(self, e2e_client: TestClient) -> None:
        """测试无效 token"""
        invalid_token = "invalid.token.here"
        headers = {"Authorization": f"Bearer {invalid_token}"}

        response = e2e_client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code in (401, 403)

    def test_session_isolation(self, e2e_client: TestClient) -> None:
        """测试会话隔离 - 不同用户的数据互不干扰"""
        # 注册两个用户
        user1_resp = e2e_client.post(
            "/api/v1/auth/register",
            json={
                "email": "user1@example.com",
                "password": "User1Pass123!",
                "full_name": "User One",
            },
        )
        token1 = user1_resp.json()["access_token"]
        headers1 = {"Authorization": f"Bearer {token1}"}

        user2_resp = e2e_client.post(
            "/api/v1/auth/register",
            json={
                "email": "user2@example.com",
                "password": "User2Pass123!",
                "full_name": "User Two",
            },
        )
        token2 = user2_resp.json()["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}

        # User1 创建会话
        session1_resp = e2e_client.post(
            "/api/v1/auth/sessions",
            data={"name": "User1 Session"},
            headers=headers1,
        )
        session1_id = session1_resp.json()["session_id"]

        # User2 创建会话
        session2_resp = e2e_client.post(
            "/api/v1/auth/sessions",
            data={"name": "User2 Session"},
            headers=headers2,
        )
        session2_id = session2_resp.json()["session_id"]

        # User1 只能看到自己的会话
        response = e2e_client.get("/api/v1/auth/sessions", headers=headers1)
        sessions1 = response.json()
        assert session1_id in [s["session_id"] for s in sessions1]
        assert session2_id not in [s["session_id"] for s in sessions1]

        # User2 只能看到自己的会话
        response = e2e_client.get("/api/v1/auth/sessions", headers=headers2)
        sessions2 = response.json()
        assert session2_id in [s["session_id"] for s in sessions2]
        assert session1_id not in [s["session_id"] for s in sessions2]

        # User1 不能删除 User2 的会话
        response = e2e_client.delete(
            f"/api/v1/auth/sessions/{session2_id}",
            headers=headers1,
        )
        assert response.status_code in (403, 404)

    def test_chat_validation(self, e2e_client: TestClient) -> None:
        """测试聊天接口参数验证"""
        # 注册并登录
        register_resp = e2e_client.post(
            "/api/v1/auth/register",
            json={
                "email": "validation@example.com",
                "password": "ValidPass123!",
                "full_name": "Validation User",
            },
        )
        token = register_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 缺少 message
        response = e2e_client.post(
            "/api/v1/chat",
            json={"session_id": "test_session"},
        )
        assert response.status_code == 422

        # 缺少 session_id
        response = e2e_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 422

        # 空 message
        response = e2e_client.post(
            "/api/v1/chat",
            json={"message": "", "session_id": "test_session"},
        )
        assert response.status_code == 422


@pytest.mark.e2e
class TestStreamingChat:
    """流式聊天测试"""

    def test_sse_content_type(self, e2e_client: TestClient) -> None:
        """测试 SSE 流式接口返回正确的 Content-Type"""
        response = e2e_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "你好",
                "session_id": "test_sse_session",
            },
        )
        # 检查 Content-Type
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_sse_stream_format(self, e2e_client: TestClient) -> None:
        """测试 SSE 流格式"""
        response = e2e_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "简单测试",
                "session_id": "test_format_session",
            },
        )

        if response.status_code == 200:
            content = response.text
            # SSE 格式应该包含 "event:" 和 "data:"
            assert "event:" in content or "data:" in content or content.strip() == ""


@pytest.mark.e2e
class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_root_endpoint(self, e2e_client: TestClient) -> None:
        """测试根路径"""
        response = e2e_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_check(self, e2e_client: TestClient) -> None:
        """测试健康检查"""
        response = e2e_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")
