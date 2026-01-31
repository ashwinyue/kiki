"""结构化输出模型定义

使用 Pydantic 模型配合 LangChain 的 with_structured_output 功能，
实现类型安全的结构化输出解析。
"""

from typing import Literal

from pydantic import BaseModel, Field

# ============== 路由决策模型 ==============


class RouteDecision(BaseModel):
    """路由决策结果

    用于 Router Agent 决定将请求路由到哪个子 Agent。
    """

    agent: str = Field(description="目标 agent 的名称，必须从可用 agent 列表中选择")
    reason: str = Field(description="选择该 agent 的原因，简要说明用户意图匹配")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="决策的置信度，0.0 到 1.0")


class SupervisorDecision(BaseModel):
    """监督决策结果

    用于 Supervisor Agent 决定下一步分配给哪个 Worker。
    """

    next: str = Field(description="下一个要执行的 worker 名称，或 'END' 表示结束")
    status: Literal["working", "done"] = Field(
        description="当前任务状态：working 表示继续工作，done 表示完成"
    )
    message: str = Field(default="", description="给用户或 worker 的消息")


class HandoffDecision(BaseModel):
    """Agent 切换决策

    用于 Handoff Agent 决定是否切换到其他 Agent。
    """

    should_handoff: bool = Field(description="是否应该切换到其他 agent")
    target_agent: str = Field(default="", description="目标 agent 名称")
    reason: str = Field(description="切换原因")


# ============== 工具调用结果模型 ==============


class ToolResult(BaseModel):
    """工具执行结果

    用于结构化返回工具执行结果。
    """

    success: bool = Field(description="工具是否执行成功")
    result: str = Field(default="", description="工具执行结果或错误信息")
    data: dict = Field(default_factory=dict, description="工具返回的额外数据")


# ============== 响应模型 ==============


class AgentResponse(BaseModel):
    """Agent 响应模型

    用于需要结构化输出的 Agent 场景。
    """

    content: str = Field(description="响应的主要内容")
    needs_tool: bool = Field(default=False, description="是否需要调用工具")
    tool_name: str = Field(default="", description="需要调用的工具名称")
    tool_args: dict = Field(default_factory=dict, description="工具调用参数")
    metadata: dict = Field(default_factory=dict, description="额外的元数据")


# ============== 分类模型 ==============


class IntentClassification(BaseModel):
    """意图分类结果

    用于用户意图识别。
    """

    intent: str = Field(description="识别出的意图类型")
    confidence: float = Field(ge=0.0, le=1.0, description="识别置信度")
    entities: dict = Field(default_factory=dict, description="提取的实体信息")


class SentimentAnalysis(BaseModel):
    """情感分析结果

    用于分析用户消息的情感倾向。
    """

    sentiment: Literal["positive", "neutral", "negative"] = Field(description="情感倾向")
    score: float = Field(ge=-1.0, le=1.0, description="情感分数，-1.0 到 1.0")
    reasoning: str = Field(default="", description="分析理由")
