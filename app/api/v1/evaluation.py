"""评估 API 路由

提供 Agent 评估接口。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request as StarletteRequest

from app.core.evaluation import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationRunner,
    create_evaluation_runner,
    get_dataset,
    list_datasets,
)
from app.core.limiter import RateLimit, limiter
from app.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ============== 存储评估结果（内存） ==============

_evaluation_results: dict[str, EvaluationReport] = {}


# ============== Request/Response Models ==============


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
    report: EvaluationReport | None = Field(None, description="评估报告")


# ============== 评估端点 ==============


@router.post("/run", response_model=EvaluationRunResponse)
@limiter.limit(RateLimit.API)
async def run_evaluation(
    request: StarletteRequest,
    data: RunEvaluationRequest,
) -> EvaluationRunResponse:
    """运行评估

    执行 Agent 评估任务。

    Args:
        request: HTTP 请求
        data: 评估请求

    Returns:
        EvaluationRunResponse: 运行结果

    Examples:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/evaluation/run",
            json={
                "dataset_name": "basic_qa",
                "evaluators": ["response_quality", "tool_call_accuracy"],
                "max_entries": 10,
            }
        )
        print(response.json())
        ```
    """
    try:
        if data.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true 已弃用，请改用 /api/v1/evaluation/run/stream",
            )
        # 检查数据集是否存在
        dataset = get_dataset(data.dataset_name)
        if dataset is None:
            available = [ds.name for ds in list_datasets()]
            raise HTTPException(
                status_code=404,
                detail=f"数据集不存在: {data.dataset_name}. 可用数据集: {available}",
            )

        # 创建评估配置
        config = EvaluationConfig(
            dataset_name=data.dataset_name,
            evaluators=data.evaluators,
            agent_type=data.agent_type,
            session_id_prefix=data.session_id_prefix,
            max_entries=data.max_entries,
            categories=data.categories,
        )

        # 创建运行器
        runner = create_evaluation_runner(config)

        # 同步执行
        result = await runner.run()

        # 存储结果
        _evaluation_results[result.run_id] = result.report

        return EvaluationRunResponse(
            run_id=result.run_id,
            status="completed",
            message=f"评估完成，通过率: {result.report.overall_pass_rate:.1%}",
        )

    except Exception as e:
        logger.exception("evaluation_run_failed", dataset=data.dataset_name)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/run/stream")
@limiter.limit(RateLimit.API)
async def run_evaluation_stream_endpoint(
    request: StarletteRequest,
    data: RunEvaluationStreamRequest,
) -> StreamingResponse:
    """运行评估（流式 SSE）"""
    # 检查数据集是否存在
    dataset = get_dataset(data.dataset_name)
    if dataset is None:
        available = [ds.name for ds in list_datasets()]
        raise HTTPException(
            status_code=404,
            detail=f"数据集不存在: {data.dataset_name}. 可用数据集: {available}",
        )

    config = EvaluationConfig(
        dataset_name=data.dataset_name,
        evaluators=data.evaluators,
        agent_type=data.agent_type,
        session_id_prefix=data.session_id_prefix,
        max_entries=data.max_entries,
        categories=data.categories,
    )

    runner = create_evaluation_runner(config)
    return await _run_evaluation_stream(runner, config)


async def _run_evaluation_stream(
    runner: EvaluationRunner,
    config: EvaluationConfig,
) -> StreamingResponse:
    """运行流式评估

    Args:
        runner: 评估运行器
        config: 评估配置

    Returns:
        StreamingResponse: SSE 流式响应
    """
    import json

    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件流"""
        run_id = str(uuid.uuid4())

        try:
            async for update in runner.run_stream(config):
                # 更新 run_id
                if update.get("type") == "start":
                    run_id = update.get("run_id", run_id)

                # 格式化 SSE
                data_str = json.dumps(update, ensure_ascii=False, default=str)
                yield f"event: {update['type']}\ndata: {data_str}\n\n"

                # 如果完成，存储结果
                if update.get("type") == "complete":
                    report_data = update.get("report")
                    if report_data:
                        from app.core.evaluation.report import EvaluationReport

                        report = EvaluationReport(**report_data)
                        _evaluation_results[run_id] = report

        except Exception as e:
            logger.exception("evaluation_stream_failed", run_id=run_id)
            error_data = {"type": "error", "error": str(e), "run_id": run_id}
            data_str = json.dumps(error_data, ensure_ascii=False)
            yield f"event: error\ndata: {data_str}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/results/{run_id}", response_model=EvaluationStatusResponse)
@limiter.limit(RateLimit.API)
async def get_evaluation_result(
    request: StarletteRequest,
    run_id: str,
) -> EvaluationStatusResponse:
    """获取评估结果

    获取指定运行的评估结果。

    Args:
        run_id: 运行 ID

    Returns:
        EvaluationStatusResponse: 评估状态和结果
    """
    report = _evaluation_results.get(run_id)

    if report is None:
        raise HTTPException(status_code=404, detail=f"评估结果不存在: {run_id}")

    return EvaluationStatusResponse(
        run_id=run_id,
        status="completed",
        report=report,
    )


@router.get("/results/{run_id}/markdown")
@limiter.limit(RateLimit.API)
async def get_evaluation_result_markdown(
    request: StarletteRequest,
    run_id: str,
) -> dict[str, str]:
    """获取评估结果（Markdown 格式）

    Args:
        run_id: 运行 ID

    Returns:
        Markdown 格式的评估报告
    """
    report = _evaluation_results.get(run_id)

    if report is None:
        raise HTTPException(status_code=404, detail=f"评估结果不存在: {run_id}")

    return {
        "run_id": run_id,
        "markdown": report.to_markdown(),
        "summary": report.get_summary(),
    }


@router.get("/results")
@limiter.limit(RateLimit.API)
async def list_evaluation_results(
    request: StarletteRequest,
    limit: int = Query(10, ge=1, le=100, description="返回数量限制"),
) -> dict[str, list[dict[str, Any]]]:
    """列出所有评估结果

    Args:
        limit: 返回数量限制

    Returns:
        评估结果列表
    """

    results = [
        {
            "run_id": run_id,
            "timestamp": report.timestamp,
            "agent_name": report.agent_name,
            "dataset_name": report.dataset_name,
            "overall_pass_rate": report.overall_pass_rate,
            "overall_score": report.overall_score,
            "total_entries": report.total_entries,
        }
        for run_id, report in list(_evaluation_results.items())[-limit:]
    ]

    return {"results": results}


@router.delete("/results/{run_id}")
@limiter.limit(RateLimit.API)
async def delete_evaluation_result(
    request: StarletteRequest,
    run_id: str,
) -> dict[str, str]:
    """删除评估结果

    Args:
        run_id: 运行 ID

    Returns:
        操作结果
    """
    if run_id not in _evaluation_results:
        raise HTTPException(status_code=404, detail=f"评估结果不存在: {run_id}")

    del _evaluation_results[run_id]

    return {"status": "success", "message": "评估结果已删除"}


# ============== 数据集端点 ==============


@router.get("/datasets", response_model=list[DatasetListItem])
@limiter.limit(RateLimit.API)
async def list_datasets_api(
    request: StarletteRequest,
) -> list[DatasetListItem]:
    """列出所有数据集

    Returns:
        数据集列表
    """
    datasets = list_datasets()

    return [
        DatasetListItem(
            name=ds.name,
            description=ds.description,
            entry_count=len(ds.entries),
            version=ds.version,
        )
        for ds in datasets
    ]


@router.get("/datasets/{dataset_name}")
@limiter.limit(RateLimit.API)
async def get_dataset_api(
    request: StarletteRequest,
    dataset_name: str,
) -> dict[str, Any]:
    """获取数据集详情

    Args:
        dataset_name: 数据集名称

    Returns:
        数据集详情
    """

    dataset = get_dataset(dataset_name)

    if dataset is None:
        available = [ds.name for ds in list_datasets()]
        raise HTTPException(
            status_code=404,
            detail=f"数据集不存在: {dataset_name}. 可用数据集: {available}",
        )

    return {
        "name": dataset.name,
        "description": dataset.description,
        "version": dataset.version,
        "metadata": dataset.metadata,
        "entries": [
            {
                "category": e.category,
                "description": e.description,
                "input_data": e.input_data,
                "expected": e.expected,
            }
            for e in dataset.entries
        ],
    }


@router.get("/datasets/{dataset_name}/entries")
@limiter.limit(RateLimit.API)
async def list_dataset_entries(
    request: StarletteRequest,
    dataset_name: str,
    category: str | None = Query(None, description="筛选类别"),
) -> dict[str, Any]:
    """列出数据集条目

    Args:
        dataset_name: 数据集名称
        category: 筛选类别

    Returns:
        条目列表
    """

    dataset = get_dataset(dataset_name)

    if dataset is None:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_name}")

    entries = dataset.entries
    if category:
        entries = [e for e in entries if e.category == category]

    return {
        "dataset_name": dataset_name,
        "total_count": len(entries),
        "entries": [
            {
                "category": e.category,
                "description": e.description,
                "input_data": e.input_data,
                "has_expected": e.expected is not None,
            }
            for e in entries
        ],
    }
