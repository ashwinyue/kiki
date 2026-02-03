"""Memory 模块

提供短期和长期记忆管理功能。

窗口记忆功能已移至 app.agent.context.sliding_window。

使用示例:
```python
from app.agent.memory import MemoryManager
from app.agent.context import SlidingContextWindow

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

# 使用滑动窗口（替代窗口记忆）
window = SlidingContextWindow(window_size=10, max_tokens=4000)
```
"""

from app.agent.memory.base import BaseLongTermMemory, BaseMemory
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.manager import MemoryManager
from app.agent.memory.short_term import ShortTermMemory

__all__ = [
    # 基础类
    "BaseMemory",
    "BaseLongTermMemory",
    # 记忆管理器
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
]
