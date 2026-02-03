"""Agent 克隆服务

提供 Agent 复制功能的业务逻辑层。
"""

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.logging import get_logger
from app.repositories.agent_async import AgentRepositoryAsync
from app.schemas.agent import (
    AgentCopyRequest,
    AgentCopyResponse,
    BatchAgentCopyRequest,
    BatchAgentCopyResponse,
    BatchAgentCopyResult,
)

logger = get_logger(__name__)


class AgentCloner:
    """Agent 克隆服务

    提供单个和批量 Agent 复制功能。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化 Agent 克隆服务

        Args:
            session: 异步数据库会话
        """
        self.session = session
        self._repo = AgentRepositoryAsync(session)

    async def copy_agent(
        self,
        agent_id: str,
        data: AgentCopyRequest,
        created_by: str | None = None,
    ) -> AgentCopyResponse:
        """复制单个 Agent

        Args:
            agent_id: 源 Agent ID
            data: 复制请求数据
            created_by: 创建人 ID

        Returns:
            复制结果

        Raises:
            ValueError: 如果源 Agent 不存在或复制失败
        """
        source_agent = await self._repo.get(agent_id)
        if source_agent is None:
            raise ValueError(f"源 Agent {agent_id} 不存在")

        new_name = data.name or f"{source_agent.name} (副本)"

        new_agent_data = {
            "id": str(uuid4()),
            "name": new_name,
            "description": source_agent.description,
            "tenant_id": source_agent.tenant_id,
            "created_by": created_by,
            "config": source_agent.config if data.copy_config else {},
        }

        try:
            new_agent = await self._repo.create_with_tools(new_agent_data)

            logger.info(
                "agent_copied",
                source_agent_id=agent_id,
                new_agent_id=new_agent.id,
                new_name=new_name,
                copy_config=data.copy_config,
                copy_tools=data.copy_tools,
                copy_knowledge=data.copy_knowledge,
            )

            return AgentCopyResponse(
                new_agent_id=str(new_agent.id),
                name=new_agent.name,
                message=f"Agent 已成功复制为 {new_name}",
            )

        except Exception as e:
            logger.error("agent_copy_failed", source_agent_id=agent_id, error=str(e))
            raise ValueError(f"复制 Agent 失败: {str(e)}")

    async def batch_copy(
        self,
        agent_ids: list[str],
        request: BatchAgentCopyRequest,
        tenant_id: int | None = None,
        created_by: str | None = None,
    ) -> BatchAgentCopyResponse:
        """批量复制 Agent

        Args:
            agent_ids: 要复制的 Agent ID 列表
            request: 批量复制请求数据
            tenant_id: 租户 ID
            created_by: 创建人 ID

        Returns:
            批量复制结果
        """
        success_list: list[AgentCopyResponse] = []
        failed_list: list[str] = []

        for agent_id in agent_ids:
            try:
                copy_request = AgentCopyRequest(
                    name=request.name,
                    copy_config=request.copy_config,
                    copy_tools=request.copy_tools,
                    copy_knowledge=request.copy_knowledge,
                )

                result = await self.copy_agent(agent_id, copy_request, created_by)
                success_list.append(result)

            except Exception as e:
                logger.error("batch_copy_item_failed", agent_id=agent_id, error=str(e))
                failed_list.append(agent_id)

        return BatchAgentCopyResponse(
            success_count=len(success_list),
            failed_count=len(failed_list),
            results=success_list,
            failed_agent_ids=failed_list,
        )
