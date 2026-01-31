"""Evaluation 模块测试

测试评估系统的核心功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.evaluation.evaluators import (
    BaseEvaluator,
    ConversationEvaluator,
    EvaluationMetric,
    EvaluationResultItem,
    ResponseEvaluator,
    ToolCallEvaluator,
    create_evaluator,
)
from app.core.evaluation.report import (
    EvaluatorSummary,
    EvaluationReport,
    create_report,
    EntryResult,
)


class TestEvaluationMetric:
    """EvaluationMetric 测试"""

    def test_creation(self) -> None:
        """测试创建指标"""
        metric = EvaluationMetric(
            name="test_metric",
            value=0.85,
            description="测试指标",
        )

        assert metric.name == "test_metric"
        assert metric.value == 0.85
        assert metric.description == "测试指标"
        assert metric.details == {}

    def test_creation_with_details(self) -> None:
        """测试创建带详细信息的指标"""
        metric = EvaluationMetric(
            name="test_metric",
            value=0.75,
            description="测试指标",
            details={"key": "value"},
        )

        assert metric.details == {"key": "value"}


class TestEvaluationResultItem:
    """EvaluationResultItem 测试"""

    def test_creation_success(self) -> None:
        """测试创建成功结果"""
        result = EvaluationResultItem(
            evaluator_name="test_evaluator",
            passed=True,
            score=0.85,
            metrics=[],
            feedback="良好",
        )

        assert result.evaluator_name == "test_evaluator"
        assert result.passed is True
        assert result.score == 0.85
        assert result.feedback == "良好"
        assert result.error is None

    def test_creation_failure(self) -> None:
        """测试创建失败结果"""
        result = EvaluationResultItem(
            evaluator_name="test_evaluator",
            passed=False,
            score=0.0,
            metrics=[],
            error="评估失败",
        )

        assert result.passed is False
        assert result.error == "评估失败"


class TestResponseEvaluator:
    """ResponseEvaluator 测试"""

    def test_initialization(self) -> None:
        """测试初始化"""
        evaluator = ResponseEvaluator()
        assert evaluator.name == "response_quality"
        assert evaluator.description == "响应质量评估器"

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self) -> None:
        """测试评估空响应"""
        evaluator = ResponseEvaluator()

        result = await evaluator.evaluate(
            input_data={"message": "测试问题"},
            output_data={"response": ""},
        )

        assert result.evaluator_name == "response_quality"
        assert result.passed is False
        assert result.score == 0.0
        assert result.feedback == "响应为空"


class TestToolCallEvaluator:
    """ToolCallEvaluator 测试"""

    def test_initialization(self) -> None:
        """测试初始化"""
        evaluator = ToolCallEvaluator()
        assert evaluator.name == "tool_call_accuracy"
        assert evaluator.description == "工具调用准确性评估器"

    @pytest.mark.asyncio
    async def test_evaluate_tool_call_success(self) -> None:
        """测试评估成功的工具调用"""
        evaluator = ToolCallEvaluator()

        result = await evaluator.evaluate(
            input_data={},
            output_data={"tool_calls": [{"name": "calculate", "args": {"expression": "2+2"}}]},
            expected={"expected_tools": ["calculate"]},
        )

        assert result.evaluator_name == "tool_call_accuracy"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_tool_call_no_expected(self) -> None:
        """测试评估没有期望工具的调用"""
        evaluator = ToolCallEvaluator()

        result = await evaluator.evaluate(
            input_data={},
            output_data={"tool_calls": [{"name": "search", "args": {"query": "test"}}]},
            expected=None,
        )

        assert result.evaluator_name == "tool_call_accuracy"

    @pytest.mark.asyncio
    async def test_evaluate_tool_call_missed(self) -> None:
        """测试评估遗漏的工具调用"""
        evaluator = ToolCallEvaluator()

        result = await evaluator.evaluate(
            input_data={},
            output_data={"tool_calls": [{"name": "search", "args": {"query": "test"}}]},
            expected={"expected_tools": ["calculate"]},
        )

        assert result.passed is False


class TestConversationEvaluator:
    """ConversationEvaluator 测试"""

    def test_initialization(self) -> None:
        """测试初始化"""
        evaluator = ConversationEvaluator()
        assert evaluator.name == "conversation_quality"
        assert evaluator.description == "对话质量评估器"

    @pytest.mark.asyncio
    async def test_evaluate_no_history(self) -> None:
        """测试评估没有历史记录的对话"""
        evaluator = ConversationEvaluator()

        result = await evaluator.evaluate(
            input_data={"message": "你好"},
            output_data={"response": "你好！有什么可以帮助你的吗？"},
            context=None,
        )

        assert result.evaluator_name == "conversation_quality"
        assert result.passed is True
        assert result.score == 0.8
        assert "无法评估连贯性" in result.feedback


class TestBaseEvaluator:
    """BaseEvaluator 抽象基类测试"""

    def test_cannot_instantiate(self) -> None:
        """测试不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore


class TestCreateEvaluator:
    """create_evaluator 工厂函数测试"""

    def test_create_response_evaluator(self) -> None:
        """测试创建响应评估器"""
        evaluator = create_evaluator("response_quality")
        assert isinstance(evaluator, ResponseEvaluator)

    def test_create_tool_call_evaluator(self) -> None:
        """测试创建工具调用评估器"""
        evaluator = create_evaluator("tool_call_accuracy")
        assert isinstance(evaluator, ToolCallEvaluator)

    def test_create_conversation_evaluator(self) -> None:
        """测试创建对话评估器"""
        evaluator = create_evaluator("conversation_quality")
        assert isinstance(evaluator, ConversationEvaluator)

    def test_create_invalid_evaluator(self) -> None:
        """测试创建无效的评估器"""
        with pytest.raises(ValueError, match="不支持的评估器类型"):
            create_evaluator("invalid_type")


class TestEntryResult:
    """EntryResult 测试"""

    def test_creation(self) -> None:
        """测试创建条目结果"""
        result = EntryResult(
            entry_id="test-123",
            category="test",
            description="测试条目",
            passed=True,
            score=0.85,
            evaluator_results=[],
        )

        assert result.entry_id == "test-123"
        assert result.category == "test"
        assert result.passed is True
        assert result.score == 0.85

    def test_creation_with_error(self) -> None:
        """测试创建带错误的条目结果"""
        result = EntryResult(
            entry_id="test-456",
            category="test",
            description="失败条目",
            passed=False,
            score=0.0,
            error="处理失败",
        )

        assert result.error == "处理失败"


class TestEvaluatorSummary:
    """EvaluatorSummary 测试"""

    def test_creation(self) -> None:
        """测试创建评估器汇总"""
        summary = EvaluatorSummary(
            name="test_evaluator",
            description="测试评估器",
            total_count=10,
            pass_count=8,
            fail_count=2,
            pass_rate=0.8,
            avg_score=0.75,
            min_score=0.5,
            max_score=1.0,
        )

        assert summary.name == "test_evaluator"
        assert summary.total_count == 10
        assert summary.pass_count == 8
        assert summary.fail_count == 2
        assert summary.pass_rate == 0.8
        assert summary.avg_score == 0.75


class TestEvaluationReport:
    """EvaluationReport 测试"""

    def test_create_report(self) -> None:
        """测试创建报告"""
        report = EvaluationReport(
            run_id="test-run-123",
            timestamp="2024-01-01T00:00:00",
            agent_name="test_agent",
            dataset_name="test_dataset",
            evaluators=["response_quality"],
            total_entries=10,
            passed_entries=8,
            failed_entries=2,
            overall_pass_rate=0.8,
            overall_score=0.75,
        )

        assert report.run_id == "test-run-123"
        assert report.agent_name == "test_agent"
        assert report.total_entries == 10
        assert report.passed_entries == 8
        assert report.failed_entries == 2
        assert report.overall_pass_rate == 0.8
        assert report.overall_score == 0.75

    def test_to_dict(self) -> None:
        """测试转换为字典"""
        report = EvaluationReport(
            run_id="test-run-123",
            timestamp="2024-01-01T00:00:00",
            agent_name="test_agent",
            dataset_name="test_dataset",
            evaluators=["response_quality"],
            total_entries=10,
            passed_entries=8,
            failed_entries=2,
            overall_pass_rate=0.8,
            overall_score=0.75,
        )

        data = report.to_dict()
        assert data["run_id"] == "test-run-123"
        assert data["total_entries"] == 10

    def test_get_summary(self) -> None:
        """测试获取摘要"""
        report = EvaluationReport(
            run_id="test-run-12345678",
            timestamp="2024-01-01T00:00:00",
            agent_name="test_agent",
            dataset_name="test_dataset",
            evaluators=["response_quality"],
            total_entries=10,
            passed_entries=8,
            failed_entries=2,
            overall_pass_rate=0.8,
            overall_score=0.75,
        )

        summary = report.get_summary()
        assert "test-run-12" in summary  # 截断的 run_id
        assert "8/10" in summary
        assert "80.0%" in summary


class TestCreateReport:
    """create_report 工厂函数测试"""

    def test_create_empty_report(self) -> None:
        """测试创建空报告"""
        report = create_report(
            run_id="test-run",
            agent_name="test_agent",
            dataset_name="test_dataset",
            evaluators=["response_quality"],
            entry_results=[],
        )

        assert report.run_id == "test-run"
        assert report.total_entries == 0
        assert report.overall_pass_rate == 0.0

    def test_create_report_with_results(self) -> None:
        """测试创建带结果的报告"""
        entry_results = [
            EntryResult(
                entry_id="1",
                category="test",
                description="测试1",
                passed=True,
                score=0.9,
                evaluator_results=[
                    {
                        "evaluator_name": "response_quality",
                        "passed": True,
                        "score": 0.9,
                    }
                ],
            ),
            EntryResult(
                entry_id="2",
                category="test",
                description="测试2",
                passed=False,
                score=0.5,
                evaluator_results=[
                    {
                        "evaluator_name": "response_quality",
                        "passed": False,
                        "score": 0.5,
                    }
                ],
            ),
        ]

        report = create_report(
            run_id="test-run",
            agent_name="test_agent",
            dataset_name="test_dataset",
            evaluators=["response_quality"],
            entry_results=entry_results,
        )

        assert report.total_entries == 2
        assert report.passed_entries == 1
        assert report.failed_entries == 1
        assert report.overall_pass_rate == 0.5
        assert report.overall_score == 0.7


@pytest.mark.parametrize("score,passed", [
    (1.0, True),
    (0.8, True),
    (0.6, True),
    (0.5, False),
    (0.0, False),
])
def test_score_pass_threshold(score: float, passed: bool) -> None:
    """参数化测试分数与通过状态的对应关系"""
    threshold = 0.6
    is_passed = score >= threshold
    assert is_passed == passed
