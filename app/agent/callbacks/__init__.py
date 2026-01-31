"""Callback Handler 模块

提供统一的 LangChain/LangGraph Callback Handler，支持：
- Langfuse 集成追踪
- Prometheus 指标收集
- 结构化日志记录

使用示例:
```python
from app.agent.callbacks import KikiCallbackHandler

handler = KikiCallbackHandler(
    session_id="session-123",
    user_id="user-456",
    enable_langfuse=True,
    enable_metrics=True,
)

config = {"callbacks": [handler]}
result = await agent.ainvoke(input_data, config)
```
"""

from app.agent.callbacks.handler import KikiCallbackHandler

__all__ = [
    "KikiCallbackHandler",
]
