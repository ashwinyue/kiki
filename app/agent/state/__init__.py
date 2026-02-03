"""状态管理模块

统一的状态定义，整合了原 state.py、state_models.py、graph/types.py。

结构：
- typeddict.py: LangGraph TypedDict 状态定义
- pydantic.py: Pydantic 验证模型
- chat.py: ChatState 及其工厂函数
- agent.py: AgentState 及其工厂函数
- react.py: ReActState 及其工厂函数
- utils.py: 工具函数

使用示例:
```python
from app.agent.state import (
    ChatState,
    AgentState,
    ReActState,
    create_chat_state,
    create_agent_state,
    should_stop_iteration,
)
```
"""

# ============== 状态类（TypedDict/MessagesState）=============
from app.agent.state.agent import (
    AgentState,
    create_agent_state,
)
from app.agent.state.chat import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_TOKENS,
    ChatState,
    create_chat_state,
    create_state_from_input,
)
from app.agent.state.react import (
    ReActState,
    create_react_state,
)

from app.agent.state.multi_agent import (
    MultiAgentState,
)

# ============== 高级状态（三层记忆架构）=============
from app.agent.state.advanced import (
    AdvancedGenerationState,
    DocumentSection,
    PlanningTree,
    MemoryContext,
    GenerationMetadata,
)

# ============== 工具函数 ==============
from app.agent.state.utils import (
    add_messages,  # 导出 add_messages 以兼容 LangGraph
    increment_iteration,
    merge_lists,
    should_stop_iteration,
)

# ============== Pydantic 验证器（可选导入）=============
# from app.agent.state.pydantic import (
#     ChatStateModel,
#     AgentStateModel,
#     ReActStateModel,
#     StateValidator,
# )

__all__ = [
    # ============== 状态类 ==============
    "ChatState",
    "AgentState",
    "ReActState",
    "MultiAgentState",
    # ============== 高级状态（三层记忆架构）=============
    "AdvancedGenerationState",
    "DocumentSection",
    "PlanningTree",
    "MemoryContext",
    "GenerationMetadata",
    # ============== 工厂函数 ==============
    "create_chat_state",
    "create_agent_state",
    "create_react_state",
    "create_state_from_input",
    # ============== 常量 ==============
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_MESSAGES",
    # ============== 工具函数 ==============
    "should_stop_iteration",
    "increment_iteration",
    "merge_lists",
    # ============== 向后兼容 ==============
    "add_messages",
]
