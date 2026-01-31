"""Memory 模块

提供短期、长期和窗口记忆管理功能。

使用示例:
```python
from app.agent.memory import MemoryManager, create_pre_model_hook
from app.llm.embeddings import DashScopeEmbeddings

# 使用 DashScope Embeddings 创建长期记忆
embeddings = DashScopeEmbeddings(
    api_key="your-dashscope-api-key",
    dimensions=1024,
)

# 创建 Memory Manager
manager = MemoryManager(session_id="session-123")

# 添加短期记忆
await manager.add_short_term_message(
    role="user",
    content="你好",
)

# 添加长期记忆
await manager.add_long_term_memory(
    content="用户偏好使用简洁的回答",
    metadata={"type": "preference"},
)

# 检索相关记忆
memories = await manager.search_long_term("用户偏好")

# 使用窗口记忆（Token 限制）
hook = create_pre_model_hook(max_tokens=384)
```
"""

from app.agent.memory.base import BaseLongTermMemory, BaseMemory
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.manager import MemoryManager
from app.agent.memory.short_term import ShortTermMemory
from app.agent.memory.window import (
    TrimStrategy,
    TokenCounterType,
    WindowMemoryManager,
    create_chat_hook,
    create_pre_model_hook,
    get_window_memory_manager,
    trim_state_messages,
)

__all__ = [
    # 基础类
    "BaseMemory",
    "BaseLongTermMemory",
    # 记忆管理器
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    # 窗口记忆
    "TrimStrategy",
    "TokenCounterType",
    "WindowMemoryManager",
    "create_pre_model_hook",
    "create_chat_hook",
    "get_window_memory_manager",
    "trim_state_messages",
]
