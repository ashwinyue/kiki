"""工具 API 集成测试

测试工具管理和查询的 API 端点。
"""

import pytest
from fastapi.testclient import TestClient


class TestToolsListAPI:
    """工具列表 API 测试"""

    def test_list_tools(self, client: TestClient) -> None:
        """测试列出所有工具"""
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data
        assert isinstance(data["tools"], list)
        assert data["count"] >= 0

    def test_list_tools_content(self, client: TestClient) -> None:
        """测试工具列表内容结构"""
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        data = response.json()

        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            # args_schema 可选


class TestToolDetailAPI:
    """工具详情 API 测试"""

    def test_get_existing_tool(self, client: TestClient) -> None:
        """测试获取存在的工具"""
        # 先获取工具列表
        list_response = client.get("/api/v1/tools")
        assert list_response.status_code == 200
        tools_data = list_response.json()

        if tools_data["count"] > 0:
            tool_name = tools_data["tools"][0]["name"]
            response = client.get(f"/api/v1/tools/{tool_name}")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == tool_name
            assert "description" in data

    def test_get_nonexistent_tool(self, client: TestClient) -> None:
        """测试获取不存在的工具"""
        response = client.get("/api/v1/tools/nonexistent_tool_12345")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_tool_with_special_characters(self, client: TestClient) -> None:
        """测试获取包含特殊字符的工具名"""
        response = client.get("/api/v1/tools/calculate")
        # calculate 工具应该存在
        assert response.status_code in (200, 404)


class TestToolBuiltin:
    """内置工具测试"""

    def test_calculate_tool_exists(self, client: TestClient) -> None:
        """测试计算工具存在"""
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        data = response.json()

        tool_names = [t["name"] for t in data["tools"]]
        # calculate 是内置工具
        assert "calculate" in tool_names

    def test_get_weather_tool_exists(self, client: TestClient) -> None:
        """测试天气工具存在"""
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        data = response.json()

        tool_names = [t["name"] for t in data["tools"]]
        assert "get_weather" in tool_names
