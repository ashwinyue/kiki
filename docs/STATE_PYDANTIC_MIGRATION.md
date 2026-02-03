# 状态管理 Pydantic 化改进

> **实施日期**: 2026-02-03
> **改进类型**: 重构 - 代码质量提升
> **状态**: ✅ 已完成

---

## 概述

将 Kiki Agent 的状态管理从纯 TypedDict 迁移到 **TypedDict + Pydantic 验证层** 的混合模式，在保持 LangGraph 兼容性的同时，获得运行时验证和更好的开发体验。

## 设计原则

### 为什么保持 TypedDict？

LangGraph 要求状态使用 TypedDict 定义（继承 `dict`），因此不能完全替换为 Pydantic。

### 混合模式方案

```
┌─────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                │
│                                                     │
│  使用 TypedDict 定义状态（框架要求）                   │
│  - ChatState(MessagesState)                         │
│  - AgentState(TypedDict)                            │
│  - ReActState(TypedDict)                            │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               Pydantic 验证层（新增）                  │
│                                                     │
│  用于验证和文档                                       │
│  - ChatStateModel                                    │
│  - AgentStateModel                                   │
│  - ReActStateModel                                   │
│                                                     │
│  提供功能：                                           │
│  - 字段验证（约束、类型检查）                           │
│  - 转换方法（to_dict, from_dict）                     │
│  - 错误提示                                           │
└─────────────────────────────────────────────────────┘
```

---

## 实施内容

### 1. 创建 Pydantic 状态模型

**文件**: `app/agent/state_models.py`

#### ChatStateModel

```python
class ChatStateModel(BaseModel):
    """ChatState 的 Pydantic 验证模型"""

    # 消息历史
    messages: list[BaseMessage] = Field(default_factory=list)

    # 用户和会话信息
    user_id: str | None = Field(default=None)
    session_id: str = Field(default="")
    tenant_id: int | None = Field(default=None)

    # 迭代控制（带约束）
    iteration_count: int = Field(default=0, ge=0, le=50)
    max_iterations: int = Field(default=10, ge=1, le=100)

    # 错误处理
    error: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_iteration_limits(self):
        """自动限制迭代次数到最大值"""
        if self.iteration_count > self.max_iterations:
            object.__setattr__(self, "iteration_count", self.max_iterations)
        return self

    def to_dict(self) -> dict[str, Any]:
        """转换为 TypedDict 兼容格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatStateModel":
        """从字典创建（支持部分数据）"""
        filtered_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls(**filtered_data)
```

#### 约束验证

| 字段 | 约束 | 说明 |
|------|------|------|
| `iteration_count` | `ge=0, le=50` | 必须在 0-50 之间 |
| `max_iterations` | `ge=1, le=100` | 必须在 1-100 之间 |
| 自动验证 | - | `iteration_count` 自动限制到 `max_iterations` |

### 2. 更新状态工厂函数

**文件**: `app/agent/state.py`, `app/agent/graph/types.py`

#### 添加验证支持

```python
# 全局配置
ENABLE_STATE_VALIDATION = True

def create_chat_state(
    messages: list[BaseMessage] | None = None,
    user_id: str | None = None,
    session_id: str = "",
    tenant_id: int | None = None,
    validate: bool | None = None,  # 新增参数
) -> ChatState:
    """创建聊天状态（可选 Pydantic 验证）"""
    state_data = {
        "messages": messages or [],
        "user_id": user_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "iteration_count": 0,
        "max_iterations": 10,
        "error": None,
    }

    # 根据配置决定是否验证
    should_validate = validate if validate is not None else ENABLE_STATE_VALIDATION

    if should_validate:
        try:
            validated_data = StateValidator.validate_chat_state(state_data)
            return ChatState(**validated_data)
        except Exception as e:
            logger.warning("chat_state_validation_failed", error=str(e))
            # 验证失败时返回未验证的状态（向后兼容）
            return ChatState(**state_data)

    return ChatState(**state_data)
```

### 3. 状态验证器

```python
class StateValidator:
    """状态验证器"""

    @staticmethod
    def validate_chat_state(data: dict[str, Any]) -> dict[str, Any]:
        """验证 ChatState 数据"""
        try:
            model = ChatStateModel.from_dict(data)
            return model.to_dict()
        except Exception as e:
            raise ValueError(f"ChatState 验证失败: {e}") from e

    @staticmethod
    def safe_validate_chat_state(data: dict[str, Any]) -> dict[str, Any] | None:
        """安全验证（不抛出异常）"""
        try:
            return StateValidator.validate_chat_state(data)
        except Exception:
            return None
```

### 4. 便捷函数增强

```python
def increment_iteration_validated(state: dict[str, Any]) -> dict[str, Any]:
    """增加迭代计数（带验证）"""
    current = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    # 检查是否超过最大值
    if current >= max_iterations:
        logger.warning("max_iterations_reached")
        return {}

    return {"iteration_count": current + 1}

def validate_state_update(
    state: dict[str, Any],
    update: dict[str, Any],
    state_type: str = "chat"
) -> dict[str, Any]:
    """验证状态更新"""
    merged = {**state, **update}

    if state_type == "chat":
        return StateValidator.validate_chat_state(merged)
    elif state_type == "agent":
        return StateValidator.validate_agent_state(merged)
    elif state_type == "react":
        return StateValidator.validate_react_state(merged)
    else:
        raise ValueError(f"不支持的状态类型: {state_type}")
```

---

## 测试

**文件**: `tests/agent/test_state_models.py`

### 测试覆盖

- ✅ 创建默认状态
- ✅ 创建带值状态
- ✅ 字段约束验证（iteration_count, max_iterations）
- ✅ 自动迭代限制
- ✅ 转换方法（to_dict, from_dict）
- ✅ 状态验证器
- ✅ 便捷函数

### 测试结果

```
=== 测试 1: 创建状态 ===
✅ 创建状态成功

=== 测试 2: 转换为字典 ===
✅ 转换成功

=== 测试 3: 验证状态 ===
✅ 验证成功

=== 测试 4: 约束验证 ===
✅ 约束验证正常: iteration_count=-1 被拒绝
✅ 约束验证正常: iteration_count=51 被拒绝

=== 测试 5: 迭代限制 ===
✅ 自动限制: iteration_count=15 被限制为 10

=== 测试 6: from_dict ===
✅ from_dict 成功

✅ 所有测试通过！
```

---

## 优势对比

| 特性 | 纯 TypedDict | TypedDict + Pydantic |
|------|-------------|---------------------|
| **LangGraph 兼容** | ✅ | ✅ |
| **运行时验证** | ❌ | ✅ |
| **字段约束** | ❌ | ✅ (ge, le 等) |
| **错误提示** | 基础 | 详细 |
| **IDE 支持** | 基础 | 优秀 |
| **自动文档** | ❌ | ✅ (自动生成) |
| **序列化** | 手动 | 自动 |
| **向后兼容** | - | ✅ (可选启用) |

---

## 使用方式

### 1. 基本使用（自动验证）

```python
from app.agent.state import create_chat_state

# 自动验证（默认启用）
state = create_chat_state(
    messages=[HumanMessage(content="Hello")],
    user_id="user123",
    session_id="session456",
)
# 如果数据无效，会记录警告并返回未验证的状态
```

### 2. 显式控制验证

```python
# 强制启用验证
state = create_chat_state(..., validate=True)

# 禁用验证（性能敏感场景）
state = create_chat_state(..., validate=False)
```

### 3. 状态更新验证

```python
from app.agent.state import validate_state_update

current_state = {"iteration_count": 5, "max_iterations": 10}
update = {"iteration_count": 6}

# 验证后的完整状态
validated = validate_state_update(current_state, update, state_type="chat")
```

### 4. 安全验证（不抛出异常）

```python
from app.agent.state_models import StateValidator

result = StateValidator.safe_validate_chat_state(data)
if result is None:
    # 验证失败，处理错误
    pass
```

---

## 配置

### 全局开关

在 `app/agent/state.py` 和 `app/agent/graph/types.py` 中：

```python
# 全局配置：是否启用 Pydantic 验证
ENABLE_STATE_VALIDATION = True
```

### 环境变量（可选扩展）

可以扩展为从环境变量读取：

```python
import os

ENABLE_STATE_VALIDATION = os.getenv("ENABLE_STATE_VALIDATION", "true").lower() == "true"
```

---

## 注意事项

### 1. 避免递归验证

在 `model_validator` 中修改字段时，使用 `object.__setattr__` 绕过 Pydantic 验证：

```python
# ❌ 错误：会触发递归
self.iteration_count = self.max_iterations

# ✅ 正确：绕过验证
object.__setattr__(self, "iteration_count", self.max_iterations)
```

### 2. 向后兼容

验证失败时不应抛出异常，而是返回未验证的状态：

```python
if should_validate:
    try:
        validated_data = StateValidator.validate_chat_state(state_data)
        return ChatState(**validated_data)
    except Exception as e:
        logger.warning("chat_state_validation_failed", error=str(e))
        # 验证失败时返回未验证的状态（向后兼容）
        return ChatState(**state_data)
```

### 3. 性能考虑

Pydantic 验证有轻微性能开销，在性能敏感场景可以禁用：

```python
# 批量操作时禁用验证
for _ in range(1000):
    state = create_chat_state(..., validate=False)
```

---

## 未来改进

1. **更多验证器**
   - 添加 `session_id` 格式验证
   - 添加 `tenant_id` 存在性检查
   - 添加 `error` 消息格式验证

2. **自定义验证器**
   - 支持用户自定义验证规则
   - 支持租户级别的验证配置

3. **性能优化**
   - 缓存验证结果
   - 延迟验证（批量验证）

4. **文档生成**
   - 自动生成状态 API 文档
   - 生成状态变更日志

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `app/agent/state_models.py` | Pydantic 状态模型定义（新增） |
| `app/agent/state.py` | 状态工厂函数（已更新） |
| `app/agent/graph/types.py` | 图状态类型（已更新） |
| `tests/agent/test_state_models.py` | 状态模型测试（新增） |

---

## 参考

- [LangGraph 状态管理](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [Pydantic 模型验证器](https://docs.pydantic.dev/latest/concepts/models/#validators)
- [架构评估报告](./AGENT_ARCHITECTURE_REVIEW.md)
