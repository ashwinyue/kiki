"""
高级 LangGraph 工作流单元测试

测试三层记忆架构的节点和工作流。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.state.advanced import AdvancedGenerationState
from app.agent.graph.advanced_builder import (
    AdvancedGenerationBuilder,
    create_advanced_generation_workflow,
)
from app.agent.graph.advanced_nodes import (
    split_node,
    summarize_and_memorize_node,
    planning_node,
    generate_node,
    split_text_semantic,
)


class TestSplitTextSemantic:
    """测试语义化文本分块"""

    def test_split_paragraphs_in_range(self):
        """测试段落数在目标范围内"""
        text = "第一段\n\n第二段\n\n第三段"
        result = split_text_semantic(text)
        assert len(result) == 3
        assert result == ["第一段", "第二段", "第三段"]

    def test_split_single_paragraph(self):
        """测试单个段落按句子分割"""
        text = "这是第一句。这是第二句。这是第三句。这是第四句。"
        result = split_text_semantic(text)
        assert len(result) >= 2
        assert any("第一句" in chunk for chunk in result)

    def test_split_many_paragraphs_merges(self):
        """测试过多段落时合并"""
        paragraphs = [f"第{i}段内容" for i in range(15)]
        text = "\n\n".join(paragraphs)
        result = split_text_semantic(text)
        assert 2 <= len(result) <= 10

    def test_split_empty_text(self):
        """测试空文本"""
        result = split_text_semantic("")
        assert result == []

    def test_split_preserves_content(self):
        """测试内容保留"""
        text = "段落A\n\n段落B"
        result = split_text_semantic(text)
        combined = "\n\n".join(result)
        assert "段落A" in combined
        assert "段落B" in combined


class TestSplitNode:
    """测试分块节点"""

    @pytest.mark.asyncio
    async def test_split_node_basic(self):
        """测试基本分块功能"""
        state: AdvancedGenerationState = {
            "original_text": "第一段\n\n第二段\n\n第三段",
            "chunks": [],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        result = await split_node(state)

        assert len(result["chunks"]) == 3
        assert result["current_step"] == "split"
        assert result["metadata"]["chunk_count"] == 3
        assert "avg_chunk_length" in result["metadata"]

    @pytest.mark.asyncio
    async def test_split_node_long_text(self):
        """测试长文本分块"""
        long_text = "\n\n".join([f"段落 {i} 的内容" for i in range(12)])
        state: AdvancedGenerationState = {
            "original_text": long_text,
            "chunks": [],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        result = await split_node(state)

        # 应该合并段落
        assert 2 <= len(result["chunks"]) <= 10


class TestSummarizeAndMemorizeNode:
    """测试摘要与记忆节点"""

    @pytest.mark.asyncio
    async def test_summarize_node_with_mock_llm(self):
        """测试使用模拟 LLM 的摘要节点"""
        state: AdvancedGenerationState = {
            "original_text": "测试文本",
            "chunks": ["第一块内容", "第二块内容"],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        # Mock LLM 服务
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=MagicMock(content="摘要内容"))

        result = await summarize_and_memorize_node(state, llm_service=mock_llm)

        assert len(result["summaries"]) == 2
        assert result["current_step"] == "summarize_and_memorize"
        assert result["metadata"]["summary_count"] == 2

    @pytest.mark.asyncio
    async def test_summarize_node_compression_ratio(self):
        """测试压缩比率计算"""
        state: AdvancedGenerationState = {
            "original_text": "测试文本" * 100,
            "chunks": ["长内容" * 50],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=MagicMock(content="短摘要"))

        result = await summarize_and_memorize_node(state, llm_service=mock_llm)

        assert "compression_ratio" in result["metadata"]
        assert 0 <= result["metadata"]["compression_ratio"] <= 1


class TestPlanningNode:
    """测试规划节点"""

    @pytest.mark.asyncio
    async def test_planning_node_creates_structure(self):
        """测试规划节点创建结构"""
        state: AdvancedGenerationState = {
            "original_text": "测试文本",
            "chunks": [],
            "summaries": ["摘要1", "摘要2", "摘要3"],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        mock_llm = MagicMock()
        mock_output = MagicMock(
            title="测试标题",
            sections=[
                {"title": "第一部分", "subsections": []},
                {"title": "第二部分", "subsections": []},
            ],
            total_paragraphs=2,
        )
        mock_llm.with_structured_output = MagicMock(return_value=AsyncMock(return_value=mock_output))

        result = await planning_node(state, llm_service=mock_llm)

        assert result["current_step"] == "planning"
        assert result["planning_tree"]["title"] == "测试标题"
        assert len(result["planning_tree"]["sections"]) == 2
        assert result["metadata"]["section_count"] == 2


class TestGenerateNode:
    """测试生成节点"""

    @pytest.mark.asyncio
    async def test_generate_node_creates_output(self):
        """测试生成节点创建输出"""
        state: AdvancedGenerationState = {
            "original_text": "测试文本",
            "chunks": [],
            "summaries": [],
            "planning_tree": {
                "title": "测试文档",
                "sections": [
                    {"title": "引言", "content": "", "subsections": []},
                    {"title": "正文", "content": "", "subsections": []},
                ],
                "total_paragraphs": 2,
            },
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=MagicMock(content="章节内容"))

        result = await generate_node(state, llm_service=mock_llm)

        assert result["current_step"] == "generate"
        assert len(result["final_output"]) > 0
        assert "测试文档" in result["final_output"]
        assert "引言" in result["final_output"]
        assert result["metadata"]["final_length"] > 0


class TestAdvancedGenerationBuilder:
    """测试高级生成构建器"""

    def test_builder_initialization(self):
        """测试构建器初始化"""
        builder = AdvancedGenerationBuilder()
        assert builder.checkpoint_saver is None

    def test_builder_with_checkpoint(self):
        """测试带检查点的构建器"""
        mock_checkpoint = MagicMock()
        builder = AdvancedGenerationBuilder(checkpoint_saver=mock_checkpoint)
        assert builder.checkpoint_saver is mock_checkpoint

    def test_build_creates_workflow(self):
        """测试构建创建工作流"""
        builder = AdvancedGenerationBuilder()
        workflow = builder.build()
        assert workflow is not None

    def test_compile_creates_runnable(self):
        """测试编译创建可运行对象"""
        builder = AdvancedGenerationBuilder()
        compiled = builder.compile()
        assert compiled is not None


class TestCreateAdvancedGenerationWorkflow:
    """测试便捷函数"""

    def test_create_workflow(self):
        """测试创建工作流"""
        workflow = create_advanced_generation_workflow()
        assert workflow is not None

    def test_create_workflow_with_checkpoint(self):
        """测试带检查点创建工作流"""
        mock_checkpoint = MagicMock()
        workflow = create_advanced_generation_workflow(checkpoint_saver=mock_checkpoint)
        assert workflow is not None


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_mocks(self):
        """测试完整工作流（使用模拟）"""
        # Mock 所有依赖
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=MagicMock(content="摘要"))
        mock_llm.with_structured_output = MagicMock(
            return_value=AsyncMock(
                return_value=MagicMock(
                    title="测试文档",
                    sections=[
                        {"title": "第一部分", "subsections": []},
                    ],
                    total_paragraphs=1,
                )
            )
        )

        # 初始状态
        initial_state: AdvancedGenerationState = {
            "original_text": "第一段\n\n第二段\n\n第三段",
            "chunks": [],
            "summaries": [],
            "planning_tree": {},
            "final_output": "",
            "vectorstore": None,
            "metadata": {},
            "current_step": "",
            "error": None,
        }

        # 执行工作流
        state = await split_node(initial_state)
        state = await summarize_and_memorize_node(state, llm_service=mock_llm)
        state = await planning_node(state, llm_service=mock_llm)
        state = await generate_node(state, llm_service=mock_llm)

        # 验证结果
        assert len(state["chunks"]) > 0
        assert len(state["summaries"]) > 0
        assert state["planning_tree"]["title"] == "测试文档"
        assert len(state["final_output"]) > 0
        assert state["current_step"] == "generate"
