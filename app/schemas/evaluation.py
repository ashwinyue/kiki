"""评估相关模式"""

from typing import Any

from pydantic import BaseModel, Field


class RunEvaluationRequest(BaseModel):
    """运行评估请求"""

    dataset_name: str = Field(
        ...,
        description="数据集名称",
        examples=["basic_qa", "tool_calls", "conversation", "edge_cases"],
    )
    evaluators: list[str] = Field(
        default=["response_quality"],
        description="评估器列表",
        examples=[["response_quality", "tool_call_accuracy"]],
    )
    agent_type: str = Field(
        default="chat",
        description="Agent 类型",
    )
    session_id_prefix: str = Field(
        default="eval-",
        description="会话 ID 前缀",
    )
    max_entries: int | None = Field(
        None,
        description="最大条目数限制",
    )
    categories: list[str] | None = Field(
        None,
        description="筛选类别",
    )
    stream: bool = Field(
        default=False,
        description="是否使用流式响应（已弃用，请改用 /run/stream）",
    )


class RunEvaluationStreamRequest(BaseModel):
    """运行评估（流式）请求"""

    dataset_name: str = Field(
        ...,
        description="数据集名称",
        examples=["basic_qa", "tool_calls", "conversation", "edge_cases"],
    )
    evaluators: list[str] = Field(
        default=["response_quality"],
        description="评估器列表",
        examples=[["response_quality", "tool_call_accuracy"]],
    )
    agent_type: str = Field(
        default="chat",
        description="Agent 类型",
    )
    session_id_prefix: str = Field(
        default="eval-",
        description="会话 ID 前缀",
    )
    max_entries: int | None = Field(
        None,
        description="最大条目数限制",
    )
    categories: list[str] | None = Field(
        None,
        description="筛选类别",
    )


class DatasetListItem(BaseModel):
    """数据集列表项"""

    name: str = Field(..., description="数据集名称")
    description: str = Field(..., description="描述")
    entry_count: int = Field(..., description="条目数量")
    version: str = Field(..., description="版本")


class EvaluationRunResponse(BaseModel):
    """评估运行响应"""

    run_id: str = Field(..., description="运行 ID")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")


class EvaluationStatusResponse(BaseModel):
    """评估状态响应"""

    run_id: str = Field(..., description="运行 ID")
    status: str = Field(..., description="状态")
    report: dict[str, Any] | None = Field(None, description="评估报告")
