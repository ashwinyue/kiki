"""Agent 系统服务

提供多 Agent 系统的存储和管理功能。
"""

from typing import Any

from fastapi import HTTPException

from app.agent.message_utils import extract_ai_content
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AgentSystemService:
    """Agent 系统服务"""

    def __init__(self) -> None:
        self._systems: dict[str, dict[str, Any]] = {}

    def store_system(
        self,
        system_type: str,
        name: str,
        graph,
        agents: list[str],
    ) -> str:
        """存储 Agent 系统

        Args:
            system_type: 系统类型
            name: 系统名称
            graph: 编译后的图
            agents: Agent 名称列表

        Returns:
            系统 ID
        """
        system_id = f"{system_type}_{name}"
        self._systems[system_id] = {
            "type": system_type,
            "name": name,
            "graph": graph,
            "agents": agents,
        }
        logger.info(
            f"{system_type}_agent_created",
            name=name,
            agents=agents,
        )
        return system_id

    def get_system(self, system_key: str) -> dict[str, Any]:
        """获取 Agent 系统

        Args:
            system_key: 系统键

        Returns:
            系统配置

        Raises:
            HTTPException: 如果系统不存在
        """
        if system_key not in self._systems:
            raise HTTPException(status_code=404, detail=f"Agent 系统 '{system_key}' 不存在")
        return self._systems[system_key]

    def list_systems(self) -> list[dict[str, Any]]:
        """列出所有 Agent 系统

        Returns:
            系统列表
        """
        return [
            {
                "id": key,
                "type": value["type"],
                "name": value["name"],
                "agents": value["agents"],
            }
            for key, value in self._systems.items()
        ]

    def delete_system(self, system_key: str) -> bool:
        """删除 Agent 系统

        Args:
            system_key: 系统键

        Returns:
            是否成功删除
        """
        if system_key in self._systems:
            del self._systems[system_key]
            logger.info("agent_system_deleted", system_key=system_key)
            return True
        return False

    async def execute_chat(
        self,
        system_id: str,
        message: str,
        session_id: str,
        user_id: str | None,
        tenant_id: int | None,
    ) -> tuple[str, str]:
        """执行 Agent 系统聊天

        Args:
            system_id: 系统 ID
            message: 用户消息
            session_id: 会话 ID
            user_id: 用户 ID
            tenant_id: 租户 ID

        Returns:
            (响应内容, session_id)
        """
        from app import get_agent
        from langgraph.types import RunnableConfig

        system = self.get_system(system_id)
        graph = system["graph"]

        # 准备输入
        from langchain_core.messages import HumanMessage

        input_data = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "session_id": session_id,
        }

        # 准备配置
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "tenant_id": tenant_id,
            },
        )

        result = await graph.ainvoke(input_data, config)
        content = extract_ai_content(result.get("messages", []))

        # 持久化交互
        agent = await get_agent()
        await agent.persist_interaction(session_id, message, content)

        return content, session_id


# 全局单例
_agent_system_service = AgentSystemService()


def get_agent_system_service() -> AgentSystemService:
    """获取 Agent 系统服务单例

    Returns:
        AgentSystemService 实例
    """
    return _agent_system_service
