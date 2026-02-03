"""实体提取器 - 增强长期记忆

从对话中提取重要实体（人物、地点、组织等），增强长期记忆能力。

参考外部项目的 Mem0 实体提取设计模式。

功能说明：
    - 从文本中提取命名实体
    - 关联实体到用户上下文
    - 支持持久化存储
    - 提供语义检索能力
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.observability.logging import get_logger

logger = get_logger(__name__)


class EntityType(str, Enum):
    """实体类型

    参考 Mem0 的实体分类。
    """

    PERSON = "person"  # 人物
    ORGANIZATION = "organization"  # 组织
    LOCATION = "location"  # 地点
    PRODUCT = "product"  # 产品
    EVENT = "event"  # 事件
    CONCEPT = "concept"  # 概念
    SKILL = "skill"  # 技能
    PREFERENCE = "preference"  # 偏好
    OTHER = "other"  # 其他


@dataclass
class Entity:
    """提取的实体

    Attributes:
        name: 实体名称
        type: 实体类型
        confidence: 置信度 (0-1)
        context: 上下文描述
        source: 来源（用户/助手）
        extracted_at: 提取时间
        metadata: 额外元数据
    """

    name: str
    type: EntityType
    confidence: float = 0.8
    context: str = ""
    source: str = "unknown"
    extracted_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "confidence": self.confidence,
            "context": self.context,
            "source": self.source,
            "extracted_at": self.extracted_at.isoformat(),
            "metadata": self.metadata,
        }


class ExtractEntitiesRequest(BaseModel):
    """实体提取请求"""

    text: str = Field(..., description="要提取实体的文本")
    user_id: str | None = Field(None, description="用户 ID")
    max_entities: int = Field(10, description="最大提取实体数")
    min_confidence: float = Field(0.5, description="最小置信度阈值")


class ExtractEntitiesResponse(BaseModel):
    """实体提取响应"""

    entities: list[Entity] = Field(default_factory=list, description="提取的实体列表")
    text_summary: str = Field("", description="文本摘要")


class EntityExtractor:
    """实体提取器

    使用 LLM 从文本中提取命名实体，增强长期记忆能力。

    设计模式参考：
    - 外部项目的 Mem0 实体提取
    - LangChain 的实体提取链
    """

    def __init__(self):
        """初始化实体提取器"""
        self._extraction_prompt = """你是专业的实体识别助手。从给定的文本中提取重要的命名实体。

请识别以下类型的实体：
- **person**: 人物（人名、昵称）
- **organization**: 组织（公司、机构）
- **location**: 地点（城市、国家、地址）
- **product**: 产品（软件、工具、服务）
- **event**: 事件（会议、项目）
- **concept**: 概念（技术、理论）
- **skill**: 技能（编程语言、框架）
- **preference**: 偏好（喜好、习惯）

要求：
1. 只提取明确提到的实体
2. 置信度 >= {min_confidence}
3. 最多提取 {max_entities} 个实体
4. 提供简短的上下文描述

返回 JSON 格式：
```json
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型",
      "confidence": 0.9,
      "context": "上下文描述"
    }}
  ],
  "text_summary": "文本摘要"
}}
```

待分析文本：
{text}"""

    async def extract(
        self,
        text: str,
        user_id: str | None = None,
        max_entities: int = 10,
        min_confidence: float = 0.5,
    ) -> ExtractEntitiesResponse:
        """从文本中提取实体

        Args:
            text: 要分析的文本
            user_id: 用户 ID
            max_entities: 最大提取实体数
            min_confidence: 最小置信度阈值

        Returns:
            提取的实体响应
        """
        try:
            from app.llm import get_llm_service

            llm_service = get_llm_service()
            llm = llm_service.get_llm()

            # 构建提示词
            prompt = self._extraction_prompt.format(
                text=text[:2000],  # 限制文本长度
                min_confidence=min_confidence,
                max_entities=max_entities,
            )

            # 调用 LLM
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])

            # 解析响应
            try:
                result = json.loads(response.content)
                entities_data = result.get("entities", [])
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，尝试提取
                entities_data = []

            # 转换为 Entity 对象
            entities = []
            for entity_data in entities_data:
                try:
                    entity = Entity(
                        name=entity_data.get("name", ""),
                        type=EntityType(entity_data.get("type", "other")),
                        confidence=entity_data.get("confidence", 0.5),
                        context=entity_data.get("context", ""),
                        source=user_id or "unknown",
                    )
                    if entity.confidence >= min_confidence:
                        entities.append(entity)
                except (ValueError, KeyError) as e:
                    logger.warning("entity_parse_failed", error=str(e))

            logger.info(
                "entities_extracted",
                count=len(entities),
                user_id=user_id,
            )

            return ExtractEntitiesResponse(
                entities=entities,
                text_summary=result.get("text_summary", ""),
            )

        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            return ExtractEntitiesResponse(entities=[], text_summary="")

    async def extract_from_messages(
        self,
        messages: list[BaseMessage],
        user_id: str | None = None,
        max_entities: int = 10,
    ) -> list[Entity]:
        """从消息列表中提取实体

        Args:
            messages: 消息列表
            user_id: 用户 ID
            max_entities: 最大提取实体数

        Returns:
            提取的实体列表
        """
        all_entities = []

        for message in messages:
            # 提取消息内容
            content = message.content if isinstance(message.content, str) else str(message.content)

            if len(content.strip()) < 10:  # 跳过太短的内容
                continue

            # 提取实体
            response = await self.extract(
                text=content,
                user_id=user_id,
                max_entities=max_entities,
            )

            all_entities.extend(response.entities)

        # 去重（按名称和类型）
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = f"{entity.name}:{entity.type.value}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        logger.info(
            "entities_extracted_from_messages",
            total=len(all_entities),
            unique=len(unique_entities),
        )

        return unique_entities


class EntityStore:
    """实体存储

    管理提取的实体，支持持久化和检索。
    """

    def __init__(self):
        """初始化实体存储"""
        self._entities: dict[str, list[Entity]] = {}  # user_id -> entities

    async def store_entities(
        self,
        user_id: str,
        entities: list[Entity],
    ) -> None:
        """存储实体

        Args:
            user_id: 用户 ID
            entities: 实体列表
        """
        if user_id not in self._entities:
            self._entities[user_id] = []

        # 添加新实体
        for entity in entities:
            # 检查是否已存在
            exists = any(
                e.name == entity.name and e.type == entity.type
                for e in self._entities[user_id]
            )
            if not exists:
                self._entities[user_id].append(entity)

        logger.info(
            "entities_stored",
            user_id=user_id,
            count=len(entities),
        )

    async def get_entities(
        self,
        user_id: str,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """获取用户的实体

        Args:
            user_id: 用户 ID
            entity_type: 实体类型（可选）

        Returns:
            实体列表
        """
        entities = self._entities.get(user_id, [])

        if entity_type:
            entities = [e for e in entities if e.type == entity_type]

        return entities

    async def search_entities(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[Entity]:
        """搜索实体

        Args:
            user_id: 用户 ID
            query: 搜索查询
            limit: 最大返回数量

        Returns:
            匹配的实体列表
        """
        entities = self._entities.get(user_id, [])

        # 简单的关键词匹配
        query_lower = query.lower()
        matches = [
            e
            for e in entities
            if query_lower in e.name.lower() or query_lower in e.context.lower()
        ]

        return matches[:limit]

    async def delete_entities(
        self,
        user_id: str,
        entity_type: EntityType | None = None,
    ) -> int:
        """删除实体

        Args:
            user_id: 用户 ID
            entity_type: 实体类型（可选，None 表示删除所有）

        Returns:
            删除的实体数量
        """
        if user_id not in self._entities:
            return 0

        if entity_type:
            original_count = len(self._entities[user_id])
            self._entities[user_id] = [
                e for e in self._entities[user_id] if e.type != entity_type
            ]
            deleted_count = original_count - len(self._entities[user_id])
        else:
            deleted_count = len(self._entities[user_id])
            del self._entities[user_id]

        logger.info(
            "entities_deleted",
            user_id=user_id,
            count=deleted_count,
        )

        return deleted_count


# ============== 全局单例 ==============

_entity_extractor = EntityExtractor()
_entity_store = EntityStore()


def get_entity_extractor() -> EntityExtractor:
    """获取实体提取器单例"""
    return _entity_extractor


def get_entity_store() -> EntityStore:
    """获取实体存储单例"""
    return _entity_store


__all__ = [
    # 数据模型
    "EntityType",
    "Entity",
    "ExtractEntitiesRequest",
    "ExtractEntitiesResponse",
    # 提取器和存储
    "EntityExtractor",
    "EntityStore",
    # 单例访问
    "get_entity_extractor",
    "get_entity_store",
]
