"""评估器模块

提供各种 Agent 评估器实现。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.llm import get_llm_service
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ============== 评估结果模型 ===============


class EvaluationMetric(BaseModel):
    """评估指标"""

    name: str = Field(..., description="指标名称")
    value: float = Field(..., description="指标值 (0-1)")
    description: str = Field(..., description="指标描述")
    details: dict[str, Any] = Field(default_factory=dict, description="详细信息")


class EvaluationResultItem(BaseModel):
    """单个评估结果"""

    evaluator_name: str = Field(..., description="评估器名称")
    passed: bool = Field(..., description="是否通过")
    score: float = Field(..., description="评分 (0-1)")
    metrics: list[EvaluationMetric] = Field(default_factory=list, description="指标列表")
    feedback: str | None = Field(None, description="反馈意见")
    error: str | None = Field(None, description="错误信息")


# ============== 评估器基类 ===============


class BaseEvaluator(ABC):
    """评估器基类

    所有评估器都应该继承此类并实现 evaluate 方法。
    """

    name: str = "base_evaluator"
    description: str = "评估器基类"

    @abstractmethod
    async def evaluate(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        expected: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResultItem:
        """执行评估

        Args:
            input_data: 输入数据
            output_data: 输出数据
            expected: 期望输出（可选）
            context: 额外上下文（可选）

        Returns:
            EvaluationResultItem: 评估结果
        """
        ...


# ============== 响应质量评估器 ===============


class ResponseEvaluator(BaseEvaluator):
    """响应质量评估器

    评估 Agent 响应的质量，包括：
    - 相关性：响应是否与输入相关
    - 准确性：响应内容是否准确
    - 完整性：响应是否完整回答了问题
    - 有用性：响应对用户是否有帮助
    """

    name = "response_quality"
    description = "响应质量评估器"

    # 评估提示词模板
    QUALITY_TEMPLATE = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个 AI 响应质量评估专家。请根据以下标准评估 AI 的响应：

评估标准：
1. 相关性 (0-1): 响应是否与用户问题直接相关
2. 准确性 (0-1): 响应内容是否准确无误
3. 完整性 (0-1): 是否完整回答了用户问题
4. 有用性 (0-1): 响应对用户是否有实际帮助
5. 清晰度 (0-1): 响应表达是否清晰易懂

请以 JSON 格式返回评估结果：
{{
    "relevance": <相关性评分>,
    "accuracy": <准确性评分>,
    "completeness": <完整性评分>,
    "helpfulness": <有用性评分>,
    "clarity": <清晰度评分>,
    "overall_score": <综合评分 0-1>,
    "feedback": "<简要反馈意见>",
    "passed": <是否通过，综合评分 >= 0.6>
}}""",
            ),
            (
                "user",
                "用户问题: {input}\n\nAI 响应: {output}\n\n期望回答: {expected}\n\n请评估此响应质量。",
            ),
        ]
    )

    def __init__(self, llm_service: Any | None = None):
        """初始化评估器

        Args:
            llm_service: LLM 服务实例
        """
        self._llm_service = llm_service or get_llm_service()
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    async def evaluate(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        expected: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResultItem:
        """评估响应质量

        Args:
            input_data: 输入数据（包含 message 等）
            output_data: 输出数据（包含 response, messages 等）
            expected: 期望输出（包含 expected_response 等）
            context: 额外上下文

        Returns:
            EvaluationResultItem: 评估结果
        """
        try:
            # 提取输入和输出
            input_text = input_data.get("message", "")
            output_text = output_data.get("response", "")
            expected_text = (expected or {}).get("expected_response", "") or "无特定要求"

            if not output_text:
                return EvaluationResultItem(
                    evaluator_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback="响应为空",
                )

            # 构建评估链
            chain = self.QUALITY_TEMPLATE | self._llm

            # 执行评估
            result = await chain.ainvoke(
                {
                    "input": input_text,
                    "output": output_text,
                    "expected": expected_text,
                }
            )

            # 解析结果
            result_data = self._parse_evaluation_result(result.content)

            # 构建指标
            metrics = [
                EvaluationMetric(
                    name="relevance",
                    value=result_data.get("relevance", 0.0),
                    description="相关性",
                ),
                EvaluationMetric(
                    name="accuracy",
                    value=result_data.get("accuracy", 0.0),
                    description="准确性",
                ),
                EvaluationMetric(
                    name="completeness",
                    value=result_data.get("completeness", 0.0),
                    description="完整性",
                ),
                EvaluationMetric(
                    name="helpfulness",
                    value=result_data.get("helpfulness", 0.0),
                    description="有用性",
                ),
                EvaluationMetric(
                    name="clarity",
                    value=result_data.get("clarity", 0.0),
                    description="清晰度",
                ),
            ]

            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=result_data.get("passed", False),
                score=result_data.get("overall_score", 0.0),
                metrics=metrics,
                feedback=result_data.get("feedback", ""),
            )

        except Exception as e:
            logger.exception("response_evaluation_failed", input_data=input_data)
            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=False,
                score=0.0,
                error=str(e),
            )

    def _parse_evaluation_result(self, result_text: str) -> dict[str, Any]:
        """解析评估结果

        Args:
            result_text: LLM 返回的结果文本

        Returns:
            解析后的结果字典
        """
        try:
            # 尝试提取 JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            return json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_evaluation_result", result_text=result_text[:200])
            # 返回默认值
            return {
                "relevance": 0.5,
                "accuracy": 0.5,
                "completeness": 0.5,
                "helpfulness": 0.5,
                "clarity": 0.5,
                "overall_score": 0.5,
                "feedback": "无法解析评估结果",
                "passed": False,
            }


# ============== 工具调用评估器 ===============


class ToolCallEvaluator(BaseEvaluator):
    """工具调用评估器

    评估 Agent 工具调用的正确性：
    - 是否调用了正确的工具
    - 工具参数是否正确
    - 工具调用次数是否合理
    """

    name = "tool_call_accuracy"
    description = "工具调用准确性评估器"

    def __init__(self):
        """初始化评估器"""
        pass

    async def evaluate(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        expected: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResultItem:
        """评估工具调用准确性

        Args:
            input_data: 输入数据
            output_data: 输出数据（包含 tool_calls）
            expected: 期望输出（包含 expected_tools）
            context: 额外上下文

        Returns:
            EvaluationResultItem: 评估结果
        """
        try:
            # 获取实际工具调用
            actual_calls = output_data.get("tool_calls", [])
            if isinstance(actual_calls, list) and len(actual_calls) > 0:
                actual_tools = [call.get("name", "") for call in actual_calls]
            else:
                actual_tools = []

            # 获取期望工具调用
            expected_tools = (expected or {}).get("expected_tools", [])
            if isinstance(expected_tools, str):
                expected_tools = [expected_tools]

            # 计算指标
            metrics = []
            passed = True
            feedback_parts = []

            # 检查是否调用了期望的工具
            if expected_tools:
                expected_found = [t for t in expected_tools if any(t in at for at in actual_tools)]
                tool_recall = len(expected_found) / len(expected_tools) if expected_tools else 1.0
                metrics.append(
                    EvaluationMetric(
                        name="tool_recall",
                        value=tool_recall,
                        description="期望工具召回率",
                    )
                )

                if tool_recall < 0.5:
                    passed = False
                    feedback_parts.append(
                        f"未调用期望的工具: {set(expected_tools) - set(expected_found)}"
                    )

            # 检查工具调用数量是否合理
            call_count = len(actual_tools)
            metrics.append(
                EvaluationMetric(
                    name="call_count",
                    value=min(call_count / 5, 1.0),  # 假设5次调用为上限
                    description="工具调用数量合理性",
                    details={"actual_count": call_count},
                )
            )

            # 综合评分
            score = sum(m.value for m in metrics) / len(metrics) if metrics else 1.0

            feedback = " ".join(feedback_parts) if feedback_parts else "工具调用符合预期"

            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=passed,
                score=score,
                metrics=metrics,
                feedback=feedback,
            )

        except Exception as e:
            logger.exception("tool_call_evaluation_failed", input_data=input_data)
            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=False,
                score=0.0,
                error=str(e),
            )


# ============== 对话质量评估器 ===============


class ConversationEvaluator(BaseEvaluator):
    """对话质量评估器

    评估多轮对话的质量：
    - 上下文连贯性
    - 响应一致性
    - 对话流程合理性
    """

    name = "conversation_quality"
    description = "对话质量评估器"

    def __init__(self, llm_service: Any | None = None):
        """初始化评估器

        Args:
            llm_service: LLM 服务实例
        """
        self._llm_service = llm_service or get_llm_service()
        self._llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    async def evaluate(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        expected: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResultItem:
        """评估对话质量

        Args:
            input_data: 输入数据
            output_data: 输出数据
            expected: 期望输出
            context: 上下文（包含对话历史）

        Returns:
            EvaluationResultItem: 评估结果
        """
        try:
            history = (context or {}).get("conversation_history", [])
            current_input = input_data.get("message", "")
            current_output = output_data.get("response", "")

            # 如果没有历史，退化为简单响应评估
            if not history:
                return EvaluationResultItem(
                    evaluator_name=self.name,
                    passed=True,
                    score=0.8,  # 默认评分
                    feedback="无对话历史，无法评估连贯性",
                    metrics=[
                        EvaluationMetric(
                            name="context_coherence",
                            value=0.8,
                            description="上下文连贯性",
                        )
                    ],
                )

            # 构建评估提示
            history_text = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in history[-5:]  # 只看最近5轮
                ]
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """评估 AI 在多轮对话中的表现：

评估标准：
1. 上下文连贯性 (0-1): 响应是否与对话历史连贯
2. 响应一致性 (0-1): 响应是否与之前信息一致
3. 对话流程 (0-1): 对话进展是否自然合理

返回 JSON 格式：
{{
    "coherence": <连贯性评分>,
    "consistency": <一致性评分>,
    "flow": <流程评分>,
    "overall_score": <综合评分>,
    "feedback": "<反馈意见>",
    "passed": <是否通过>
}}""",
                    ),
                    ("user", "对话历史:\n{history}\n\n当前输入: {input}\n\n当前响应: {output}"),
                ]
            )

            chain = prompt | self._llm

            result = await chain.ainvoke(
                {
                    "history": history_text,
                    "input": current_input,
                    "output": current_output,
                }
            )

            result_data = self._parse_evaluation_result(result.content)

            metrics = [
                EvaluationMetric(
                    name="coherence",
                    value=result_data.get("coherence", 0.0),
                    description="上下文连贯性",
                ),
                EvaluationMetric(
                    name="consistency",
                    value=result_data.get("consistency", 0.0),
                    description="响应一致性",
                ),
                EvaluationMetric(
                    name="flow", value=result_data.get("flow", 0.0), description="对话流程"
                ),
            ]

            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=result_data.get("passed", True),
                score=result_data.get("overall_score", 0.0),
                metrics=metrics,
                feedback=result_data.get("feedback", ""),
            )

        except Exception as e:
            logger.exception("conversation_evaluation_failed", input_data=input_data)
            return EvaluationResultItem(
                evaluator_name=self.name,
                passed=False,
                score=0.0,
                error=str(e),
            )

    def _parse_evaluation_result(self, result_text: str) -> dict[str, Any]:
        """解析评估结果"""
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {
                "coherence": 0.7,
                "consistency": 0.7,
                "flow": 0.7,
                "overall_score": 0.7,
                "feedback": "无法解析评估结果",
                "passed": True,
            }


# ============== 便捷函数 ===============


def create_evaluator(
    evaluator_type: str,
    **kwargs: Any,
) -> BaseEvaluator:
    """创建评估器实例

    Args:
        evaluator_type: 评估器类型 (response_quality, tool_call_accuracy, conversation_quality)
        **kwargs: 额外参数

    Returns:
        评估器实例

    Raises:
        ValueError: 不支持的评估器类型
    """
    evaluators: dict[str, type[BaseEvaluator]] = {
        "response_quality": ResponseEvaluator,
        "tool_call_accuracy": ToolCallEvaluator,
        "conversation_quality": ConversationEvaluator,
    }

    evaluator_class = evaluators.get(evaluator_type)
    if evaluator_class is None:
        raise ValueError(f"不支持的评估器类型: {evaluator_type}")

    return evaluator_class(**kwargs)
