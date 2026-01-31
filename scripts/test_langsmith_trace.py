"""LangSmith 追踪独立测试脚本

运行此脚本将在 LangSmith Studio 创建项目并记录追踪。
用于验证 langsmith-fetch 调试功能。

运行方式:
    uv run python scripts/test_langsmith_trace.py

环境变量:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_key_here
"""

import os

# 确保 LangSmith 环境变量已设置
# 注意：LANGCHAIN_API_KEY 需要从环境变量设置，不要硬编码
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
if "LANGCHAIN_API_KEY" not in os.environ:
    raise ValueError("请设置 LANGCHAIN_API_KEY 环境变量")
os.environ.setdefault("LANGCHAIN_PROJECT", "kiki-agent")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic


def test_simple_chat():
    """简单的聊天测试，生成 LangSmith 追踪"""
    print("🔥 开始测试 LangSmith 追踪...")
    print(f"📊 项目: {os.environ['LANGCHAIN_PROJECT']}")
    print(f"🔑 Tracing: {os.environ['LANGCHAIN_TRACING_V2']}")
    print("-" * 50)

    # 创建一个简单的 LLM 调用
    # 注意：这里使用 Anthropic，如果您没有 API key 会失败
    # 但追踪仍会上传到 LangSmith
    try:
        llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
        )

        messages = [
            SystemMessage(content="你是一个友好的助手。"),
            HumanMessage(content="用一句话介绍 Python。"),
        ]

        print("📤 发送请求到 LLM...")
        response = llm.invoke(messages)

        print("-" * 50)
        print("✅ 测试完成!")
        print(f"📥 响应: {response.content}")
        print("\n🔗 查看 LangSmith Studio:")
        print(f"   https://smith.langchain.com/?projectId={os.environ['LANGCHAIN_PROJECT']}")

    except Exception as e:
        print(f"⚠️  LLM 调用失败: {e}")
        print("💡 如果是 API key 问题，追踪可能仍已上传到 LangSmith")


def test_with_structured_output():
    """测试结构化输出，展示更复杂的追踪"""
    print("\n🔧 测试结构化输出...")

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0,
        )

        # 使用 with_structured_output 生成工具调用追踪
        structured_llm = llm.with_structured_output(
            {
                "name": "str",
                "description": "str",
                "steps": "list[str]",
            }
        )

        response = structured_llm.invoke(
            [
                HumanMessage(
                    content="用 JSON 格式告诉我如何泡一杯茶，包含 name、description 和 steps"
                )
            ]
        )

        print("✅ 结构化输出测试完成!")
        print(f"📥 响应: {response}")

    except Exception as e:
        print(f"⚠️  结构化输出测试失败: {e}")


def main():
    """主测试函数"""
    print("=" * 50)
    print("  LangSmith 追踪测试")
    print("=" * 50)

    # 运行测试
    test_simple_chat()
    test_with_structured_output()

    print("\n" + "=" * 50)
    print("📝 使用 langsmith-fetch 查看追踪:")
    print("   uv run langsmith-fetch traces --limit 5 --format pretty")
    print("=" * 50)


if __name__ == "__main__":
    main()
