"""长文本处理模块

提供 Token 计算和智能截断功能，处理长对话历史和长文档。
支持中英文 Token 计算，自动压缩和保留重要信息。

模块化结构：
- token_counter: Token 计算功能
- text_truncation: 文本截断功能
- compressor: 上下文压缩功能
- manager: 上下文管理器
- sliding_window: 滑动窗口管理

使用示例:
```python
from app.agent.context import (
    count_tokens,
    truncate_messages,
    compress_context,
    ContextManager,
    SlidingContextWindow,
)

# 计算 Token 数
token_count = count_tokens("Hello, world!", model="gpt-4o")

# 截断消息列表
truncated = truncate_messages(messages, max_tokens=4000)

# 压缩上下文
compressed = await compress_context(messages, target_tokens=2000)

# 使用上下文管理器
manager = ContextManager(max_tokens=8000)
manager.add_messages(messages)
await manager.optimize()

# 使用滑动窗口
window = SlidingContextWindow(window_size=10, max_tokens=4000)
window.add(message)
messages = window.get_messages()
```
"""

# ============== Token 计算 ==============
# ============== 上下文压缩 ==============
from app.agent.context.compressor import (
    ContextCompressor,
    compress_context,
)

# ============== 上下文管理器 ==============
from app.agent.context.manager import (
    ContextManager,
)

# ============== 滑动窗口 ==============
from app.agent.context.sliding_window import (
    SlidingContextWindow,
)

# ============== 文本截断 ==============
from app.agent.context.text_truncation import (
    truncate_messages,
    truncate_text,
)
from app.agent.context.token_counter import (
    _get_encoding,
    _tiktoken_available,
    count_messages_tokens,
    count_tokens,
    count_tokens_precise,
)

__all__ = [
    # Token 计算
    "count_tokens",
    "count_messages_tokens",
    "count_tokens_precise",
    # TikToken
    "_tiktoken_available",
    "_get_encoding",
    # 文本截断
    "truncate_messages",
    "truncate_text",
    # 上下文压缩
    "ContextCompressor",
    "compress_context",
    # 上下文管理器
    "ContextManager",
    # 滑动窗口
    "SlidingContextWindow",
]
