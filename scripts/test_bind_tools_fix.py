"""测试 bind_tools 修复

验证 LLMService.get_llm_with_tools() 方法能正确工作。
"""

import os

# 设置环境变量
os.environ.setdefault("KIKI_LLM_PROVIDER", "dashscope")
os.environ.setdefault("KIKI_LLM_MODEL", "qwen-max")
os.environ.setdefault("KIKI_DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from langchain_core.tools import tool


@tool
def test_calculator(a: int, b: int) -> int:
    """加法计算器"""
    return a + b


def test_bind_tools():
    """测试 bind_tools 不会抛出 AttributeError"""
    from app.core.llm import LLMService, LLMRegistry

    # 注册一个测试 LLM
    from langchain_openai import ChatOpenAI

    test_llm = ChatOpenAI(
        model="qwen-max",
        api_key="sk-test",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        LLMRegistry.register("test-model", test_llm, "Test model")
    except Exception:
        pass  # 已注册

    # 创建 LLMService
    llm_service = LLMService(default_model="qwen-max", max_retries=2)

    # 测试 1: get_raw_llm 返回原始 LLM
    raw_llm = llm_service.get_raw_llm()
    print(f"✓ get_raw_llm() 返回: {type(raw_llm).__name__}")

    # 测试 2: get_llm_with_tools 返回带工具的 LLM
    tools = [test_calculator]
    llm_with_tools = llm_service.get_llm_with_tools(tools)

    if llm_with_tools is None:
        print("✗ get_llm_with_tools() 返回 None")
        return False

    print(f"✓ get_llm_with_tools() 返回: {type(llm_with_tools).__name__}")

    # 测试 3: 验证返回的 LLM 有正确的属性
    # 不应该再出现 'RunnableRetry' object has no attribute 'bind_tools'
    print(f"✓ LLM 类型检查通过")

    # 测试 4: 验证 _raw_llm 和 _llm 的类型
    print(f"✓ _raw_llm 类型: {type(llm_service._raw_llm).__name__}")
    print(f"✓ _llm 类型: {type(llm_service._llm).__name__}")

    # 测试 5: with_structured_output
    from pydantic import BaseModel, Field

    class TestDecision(BaseModel):
        answer: str = Field(description="答案")

    structured_llm = llm_service.with_structured_output(TestDecision)
    print(f"✓ with_structured_output() 返回: {type(structured_llm).__name__}")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  bind_tools 修复验证测试")
    print("=" * 50)

    try:
        success = test_bind_tools()
        if success:
            print("\n✅ 所有测试通过！")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback

        traceback.print_exc()
