"""Agent 仓储模块

提供 Agent、Tool、Memory 等数据访问层。
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session, select
from sqlalchemy import desc

from app.models.agent import (
    Agent,
    AgentCreate,
    AgentUpdate,
    Tool,
    ToolCreate,
    ToolUpdate,
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
    AgentTool,
    AgentExecution,
    AgentExecutionCreate,
    AgentType,
    AgentStatus,
)
from app.models.database import (
    Memory,
    MemoryCreate,
    MemoryUpdate,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


# ============== Agent 仓储 ==============

class AgentRepository:
    """Agent 仓储

    提供 Agent 配置的 CRUD 操作。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: AgentCreate) -> Agent:
        """创建 Agent

        Args:
            data: Agent 创建数据

        Returns:
            创建的 Agent
        """
        agent = Agent(**data.model_dump(exclude={"tool_ids"}))
        self._session.add(agent)
        self._session.flush()

        # 关联工具
        if data.tool_ids:
            for tool_id in data.tool_ids:
                agent_tool = AgentTool(agent_id=agent.id, tool_id=tool_id)
                self._session.add(agent_tool)

        logger.info("agent_created", agent_id=agent.id, name=agent.name)
        return agent

    def get(self, agent_id: int) -> Agent | None:
        """获取 Agent

        Args:
            agent_id: Agent ID

        Returns:
            Agent 实例或 None
        """
        return self._session.get(Agent, agent_id)

    def get_by_name(self, name: str) -> Agent | None:
        """根据名称获取 Agent

        Args:
            name: Agent 名称

        Returns:
            Agent 实例或 None
        """
        statement = select(Agent).where(Agent.name == name)
        return self._session.exec(statement).first()

    def list(
        self,
        agent_type: AgentType | None = None,
        status: AgentStatus | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Agent]:
        """列出 Agent

        Args:
            agent_type: 筛选类型
            status: 筛选状态
            offset: 偏移量
            limit: 限制数量

        Returns:
            Agent 列表
        """
        statement = select(Agent)

        if agent_type:
            statement = statement.where(Agent.agent_type == agent_type)
        if status:
            statement = statement.where(Agent.status == status)

        statement = statement.offset(offset).limit(limit).order_by(desc(Agent.created_at))
        return self._session.exec(statement).all()

    def update(self, agent: Agent, data: AgentUpdate) -> Agent:
        """更新 Agent

        Args:
            agent: Agent 实例
            data: 更新数据

        Returns:
            更新后的 Agent
        """
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(agent, field, value)

        self._session.flush()
        logger.info("agent_updated", agent_id=agent.id)
        return agent

    def delete(self, agent: Agent) -> None:
        """删除 Agent

        Args:
            agent: Agent 实例
        """
        # 软删除
        agent.status = AgentStatus.DELETED
        self._session.flush()
        logger.info("agent_deleted", agent_id=agent.id)

    def add_tool(self, agent: Agent, tool_id: int, enabled: bool = True) -> AgentTool:
        """为 Agent 添加工具

        Args:
            agent: Agent 实例
            tool_id: 工具 ID
            enabled: 是否启用

        Returns:
            AgentTool 关联实例
        """
        agent_tool = AgentTool(
            agent_id=agent.id,
            tool_id=tool_id,
            enabled=enabled,
        )
        self._session.add(agent_tool)
        self._session.flush()
        logger.info("tool_added_to_agent", agent_id=agent.id, tool_id=tool_id)
        return agent_tool

    def remove_tool(self, agent: Agent, tool_id: int) -> None:
        """从 Agent 移除工具

        Args:
            agent: Agent 实例
            tool_id: 工具 ID
        """
        statement = select(AgentTool).where(
            AgentTool.agent_id == agent.id,
            AgentTool.tool_id == tool_id,
        )
        agent_tool = self._session.exec(statement).first()
        if agent_tool:
            self._session.delete(agent_tool)
            self._session.flush()
            logger.info("tool_removed_from_agent", agent_id=agent.id, tool_id=tool_id)

    def get_tools(self, agent: Agent) -> list[Tool]:
        """获取 Agent 的工具列表

        Args:
            agent: Agent 实例

        Returns:
            工具列表
        """
        statement = (
            select(Tool)
            .join(AgentTool)
            .where(AgentTool.agent_id == agent.id, AgentTool.enabled == True)
        )
        return self._session.exec(statement).all()


# ============== Tool 仓储 ==============

class ToolRepository:
    """工具仓储"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: ToolCreate) -> Tool:
        """创建工具"""
        tool = Tool(**data.model_dump())
        self._session.add(tool)
        self._session.flush()
        logger.info("tool_created", tool_id=tool.id, name=tool.name)
        return tool

    def get(self, tool_id: int) -> Tool | None:
        """获取工具"""
        return self._session.get(Tool, tool_id)

    def get_by_name(self, name: str) -> Tool | None:
        """根据名称获取工具"""
        statement = select(Tool).where(Tool.name == name)
        return self._session.exec(statement).first()

    def list(
        self,
        is_active: bool | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Tool]:
        """列出工具"""
        statement = select(Tool)

        if is_active is not None:
            statement = statement.where(Tool.is_active == is_active)

        statement = statement.offset(offset).limit(limit)
        return self._session.exec(statement).all()

    def update(self, tool: Tool, data: ToolUpdate) -> Tool:
        """更新工具"""
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(tool, field, value)
        self._session.flush()
        logger.info("tool_updated", tool_id=tool.id)
        return tool


# ============== Memory 仓储（LangGraph Store）=============

class MemoryRepository:
    """记忆仓储

    实现 LangGraph Store 接口，用于长期记忆存储。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def put(self, namespace: str, key: str, value: dict) -> None:
        """存储记忆

        Args:
            namespace: 命名空间
            key: 键
            value: 值（JSON）
        """
        # 查找是否已存在
        statement = select(Memory).where(
            Memory.namespace == namespace,
            Memory.key == key,
        )
        memory = self._session.exec(statement).first()

        if memory:
            memory.value = value
        else:
            memory = Memory(namespace=namespace, key=key, value=value)
            self._session.add(memory)

        self._session.flush()
        logger.debug("memory_stored", namespace=namespace, key=key)

    def get(self, namespace: str, key: str) -> dict | None:
        """获取记忆

        Args:
            namespace: 命名空间
            key: 键

        Returns:
            记忆值或 None
        """
        statement = select(Memory).where(
            Memory.namespace == namespace,
            Memory.key == key,
        )
        memory = self._session.exec(statement).first()
        return memory.value if memory else None

    def list(self, namespace: str) -> list[Memory]:
        """列出命名空间下的所有记忆

        Args:
            namespace: 命名空间

        Returns:
            记忆列表
        """
        statement = select(Memory).where(Memory.namespace == namespace)
        return self._session.exec(statement).all()

    def delete(self, namespace: str, key: str) -> bool:
        """删除记忆

        Args:
            namespace: 命名空间
            key: 键

        Returns:
            是否删除成功
        """
        statement = select(Memory).where(
            Memory.namespace == namespace,
            Memory.key == key,
        )
        memory = self._session.exec(statement).first()
        if memory:
            self._session.delete(memory)
            self._session.flush()
            logger.debug("memory_deleted", namespace=namespace, key=key)
            return True
        return None


# ============== PromptTemplate 仓储 ==============

class PromptTemplateRepository:
    """Prompt 模板仓储"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: PromptTemplateCreate) -> PromptTemplate:
        """创建模板"""
        template = PromptTemplate(**data.model_dump())
        self._session.add(template)
        self._session.flush()
        logger.info("prompt_template_created", template_id=template.id, name=template.name)
        return template

    def get(self, template_id: int) -> PromptTemplate | None:
        """获取模板"""
        return self._session.get(PromptTemplate, template_id)

    def get_by_name(self, name: str) -> PromptTemplate | None:
        """根据名称获取模板"""
        statement = select(PromptTemplate).where(PromptTemplate.name == name)
        return self._session.exec(statement).first()

    def list(
        self,
        category: str | None = None,
        is_active: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> list[PromptTemplate]:
        """列出模板"""
        statement = select(PromptTemplate).where(PromptTemplate.is_active == is_active)

        if category:
            statement = statement.where(PromptTemplate.category == category)

        statement = statement.offset(offset).limit(limit)
        return self._session.exec(statement).all()

    def update(self, template: PromptTemplate, data: PromptTemplateUpdate) -> PromptTemplate:
        """更新模板"""
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(template, field, value)
        self._session.flush()
        logger.info("prompt_template_updated", template_id=template.id)
        return template


# ============== AgentExecution 仓储 ==============

class AgentExecutionRepository:
    """Agent 执行历史仓储"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, data: AgentExecutionCreate) -> AgentExecution:
        """创建执行记录"""
        execution = AgentExecution(**data.model_dump())
        self._session.add(execution)
        self._session.flush()
        return execution

    def get(self, execution_id: int) -> AgentExecution | None:
        """获取执行记录"""
        return self._session.get(AgentExecution, execution_id)

    def list_by_session(
        self,
        session_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[AgentExecution]:
        """列出会话的执行记录"""
        statement = select(AgentExecution).where(
            AgentExecution.session_id == session_id
        ).order_by(desc(AgentExecution.created_at))

        statement = statement.offset(offset).limit(limit)
        return self._session.exec(statement).all()

    def list_by_agent(
        self,
        agent_id: int,
        offset: int = 0,
        limit: int = 100,
    ) -> list[AgentExecution]:
        """列出 Agent 的执行记录"""
        statement = select(AgentExecution).where(
            AgentExecution.agent_id == agent_id
        ).order_by(desc(AgentExecution.created_at))

        statement = statement.offset(offset).limit(limit)
        return self._session.exec(statement).all()


# ============== LangGraph Store 适配器 ==============

class StoreAdapter:
    """LangGraph Store 适配器

    将 MemoryRepository 适配为 LangGraph Store 接口。
    LangGraph Store 用于存储和检索跨会话的记忆。
    """

    def __init__(self, session: Session) -> None:
        self._memory_repo = MemoryRepository(session)

    async def aget(self, namespace: str, key: str) -> Any | None:
        """异步获取记忆"""
        return self._memory_repo.get(namespace, key)

    async def aput(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """异步存储记忆"""
        self._memory_repo.put(namespace, key, value)

    async def adelete(self, namespace: str, key: str) -> None:
        """异步删除记忆"""
        self._memory_repo.delete(namespace, key)

    async def alist(self, namespace: str) -> list[tuple[str, Any]]:
        """异步列出命名空间下的所有记忆"""
        memories = self._memory_repo.list(namespace)
        return [(m.key, m.value) for m in memories]
