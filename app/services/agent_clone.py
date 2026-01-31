"""Agent 克隆服务

提供 Agent 深度复制功能，包括配置、工具关联、知识库关联等。
对齐 WeKnora99 的 Agent Copy API。
"""

from copy import deepcopy
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.custom_agent import CustomAgent
from app.observability.logging import get_logger
from app.repositories.agent_async import AgentRepositoryAsync
from app.schemas.agent import (
    AgentCopyRequest,
    AgentCopyResponse,
    BatchAgentCopyRequest,
    BatchAgentCopyResponse,
)

logger = get_logger(__name__)


class AgentCloner:
    """Agent 克隆器

    提供 Agent 的深度复制功能，支持：
    - 基础配置复制
    - 工具关联复制
    - 知识库关联复制
    - 批量复制
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化 Agent 克隆器

        Args:
            session: 异步数据库会话
        """
        self.session = session
        self.repository = AgentRepositoryAsync(session)

    async def copy_agent(
        self,
        source_agent_id: str,
        request: AgentCopyRequest,
        created_by: str | None = None,
    ) -> AgentCopyResponse:
        """复制单个 Agent

        Args:
            source_agent_id: 源 Agent ID
            request: 复制请求
            created_by: 创建人 ID

        Returns:
            复制响应

        Raises:
            ValueError: 源 Agent 不存在
        """
        # 获取源 Agent
        source_agent = await self.repository.get(source_agent_id)
        if source_agent is None:
            raise ValueError(f"源 Agent {source_agent_id} 不存在")

        # 确定新配置
        new_config = {}
        if request.copy_config:
            # 深度复制配置，避免引用共享
            new_config = deepcopy(source_agent.config or {})
            # 清除可能需要重置的字段
            new_config.pop("id", None)
            new_config.pop("created_at", None)

        # 创建新 Agent
        new_agent = await self.repository.copy(
            source_agent_id=source_agent_id,
            new_name=request.name,
            new_description=request.description,
            new_config=new_config,
            created_by=created_by,
        )

        if new_agent is None:
            raise ValueError(f"复制 Agent {source_agent_id} 失败")

        # 复制工具关联
        copied_tools: list[str] = []
        if request.copy_tools:
            copied_tools = await self._copy_tools(source_agent_id, new_agent.id)

        # 复制知识库关联
        copied_knowledge: list[str] = []
        if request.copy_knowledge:
            copied_knowledge = await self._copy_knowledge(source_agent_id, new_agent.id)

        logger.info(
            "agent_copy_success",
            source_agent_id=source_agent_id,
            new_agent_id=new_agent.id,
            copied_tools_count=len(copied_tools),
            copied_knowledge_count=len(copied_knowledge),
        )

        return AgentCopyResponse(
            source_agent_id=source_agent_id,
            new_agent_id=str(new_agent.id),
            name=new_agent.name,
            description=new_agent.description,
            config=new_agent.config or {},
            copied_tools=copied_tools,
            copied_knowledge=copied_knowledge,
        )

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
            request: 批量复制请求
            tenant_id: 租户 ID（用于名称冲突检查）
            created_by: 创建人 ID

        Returns:
            批量复制响应
        """
        results: list[AgentCopyResponse] = []
        errors: dict[str, str] = {}

        for agent_id in agent_ids:
            try:
                # 获取源 Agent
                source_agent = await self.repository.get(agent_id)
                if source_agent is None:
                    errors[agent_id] = "Agent 不存在"
                    continue

                # 租户检查
                if tenant_id is not None and source_agent.tenant_id != tenant_id:
                    errors[agent_id] = "无权访问此 Agent"
                    continue

                # 生成新名称
                new_name = self._generate_copy_name(
                    source_agent.name,
                    request.name_suffix,
                    tenant_id,
                )

                # 构建复制请求
                copy_request = AgentCopyRequest(
                    name=new_name,
                    description=source_agent.description,
                    copy_config=request.copy_config,
                    copy_tools=request.copy_tools,
                    copy_knowledge=request.copy_knowledge,
                )

                # 执行复制
                result = await self.copy_agent(agent_id, copy_request, created_by)
                results.append(result)

            except Exception as e:
                logger.error("batch_copy_agent_failed", agent_id=agent_id, error=str(e))
                errors[agent_id] = str(e)

        return BatchAgentCopyResponse(
            success_count=len(results),
            failed_count=len(errors),
            results=results,
            errors=errors,
        )

    async def _copy_tools(self, source_agent_id: str, new_agent_id: str) -> list[str]:
        """复制工具关联

        Args:
            source_agent_id: 源 Agent ID
            new_agent_id: 新 Agent ID

        Returns:
            复制的工具 ID 列表
        """
        try:
            # 查询源 Agent 的工具关联
            from sqlalchemy import select

            # 假设有 agent_tools 关联表
            # 这里需要根据实际表结构调整
            # 如果有 agent_tools 表，执行复制操作

            # TODO: 实现工具关联复制
            # 当前返回空列表，待实现具体逻辑
            logger.info(
                "copy_tools",
                source_agent_id=source_agent_id,
                new_agent_id=new_agent_id,
                count=0,
            )
            return []

        except Exception as e:
            logger.error("copy_tools_failed", source_agent_id=source_agent_id, error=str(e))
            return []

    async def _copy_knowledge(self, source_agent_id: str, new_agent_id: str) -> list[str]:
        """复制知识库关联

        Args:
            source_agent_id: 源 Agent ID
            new_agent_id: 新 Agent ID

        Returns:
            复制的知识库 ID 列表
        """
        try:
            # 查询源 Agent 的知识库关联
            # 这里需要根据实际表结构调整

            # TODO: 实现知识库关联复制
            # 当前返回空列表，待实现具体逻辑
            logger.info(
                "copy_knowledge",
                source_agent_id=source_agent_id,
                new_agent_id=new_agent_id,
                count=0,
            )
            return []

        except Exception as e:
            logger.error(
                "copy_knowledge_failed", source_agent_id=source_agent_id, error=str(e)
            )
            return []

    def _generate_copy_name(
        self,
        original_name: str,
        suffix: str,
        tenant_id: int | None = None,
    ) -> str:
        """生成复制后的名称

        Args:
            original_name: 原始名称
            suffix: 名称后缀
            tenant_id: 租户 ID

        Returns:
            新名称
        """
        base_name = original_name
        counter = 1

        # 如果已经有后缀，先移除旧的计数器
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]

        # 尝试生成唯一名称
        while counter < 100:
            if counter == 1:
                new_name = f"{base_name}{suffix}"
            else:
                new_name = f"{base_name}{suffix} {counter}"

            # 检查名称是否已存在（简单检查，实际可能需要数据库查询）
            # 这里简化处理，直接返回
            return new_name

            counter += 1

        # 降级：添加 UUID
        import uuid

        return f"{base_name}{suffix}_{uuid.uuid4().hex[:8]}"

    async def get_copy_preview(self, agent_id: str) -> dict[str, Any]:
        """获取复制预览

        展示复制后将会包含的内容。

        Args:
            agent_id: Agent ID

        Returns:
            预览信息
        """
        agent = await self.repository.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} 不存在")

        # 获取关联的工具
        tools = await self._get_agent_tools(agent_id)

        # 获取关联的知识库
        knowledge = await self._get_agent_knowledge(agent_id)

        return {
            "agent": {
                "id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "config": agent.config,
            },
            "tools": tools,
            "knowledge": knowledge,
            "estimated_copy_size": self._estimate_copy_size(agent.config, tools, knowledge),
        }

    async def _get_agent_tools(self, agent_id: str) -> list[dict[str, Any]]:
        """获取 Agent 关联的工具

        Args:
            agent_id: Agent ID

        Returns:
            工具列表
        """
        # TODO: 实现获取工具关联逻辑
        return []

    async def _get_agent_knowledge(self, agent_id: str) -> list[dict[str, Any]]:
        """获取 Agent 关联的知识库

        Args:
            agent_id: Agent ID

        Returns:
            知识库列表
        """
        # TODO: 实现获取知识库关联逻辑
        return []

    def _estimate_copy_size(
        self,
        config: dict[str, Any],
        tools: list[dict[str, Any]],
        knowledge: list[dict[str, Any]],
    ) -> int:
        """估算复制大小

        Args:
            config: 配置
            tools: 工具列表
            knowledge: 知识库列表

        Returns:
            估算的字节数
        """
        import json

        size = len(json.dumps(config))
        size += len(json.dumps(tools))
        size += len(json.dumps(knowledge))
        return size


# 便捷函数


async def copy_agent(
    session: AsyncSession,
    source_agent_id: str,
    request: AgentCopyRequest,
    created_by: str | None = None,
) -> AgentCopyResponse:
    """复制 Agent（便捷函数）

    Args:
        session: 数据库会话
        source_agent_id: 源 Agent ID
        request: 复制请求
        created_by: 创建人 ID

    Returns:
        复制响应
    """
    cloner = AgentCloner(session)
    return await cloner.copy_agent(source_agent_id, request, created_by)


async def batch_copy_agents(
    session: AsyncSession,
    agent_ids: list[str],
    request: BatchAgentCopyRequest,
    tenant_id: int | None = None,
    created_by: str | None = None,
) -> BatchAgentCopyResponse:
    """批量复制 Agent（便捷函数）

    Args:
        session: 数据库会话
        agent_ids: 要复制的 Agent ID 列表
        request: 批量复制请求
        tenant_id: 租户 ID
        created_by: 创建人 ID

    Returns:
        批量复制响应
    """
    cloner = AgentCloner(session)
    return await cloner.batch_copy(agent_ids, request, tenant_id, created_by)
