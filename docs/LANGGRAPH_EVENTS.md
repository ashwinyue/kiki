# LangGraph 事件处理

## 概述

Kiki **不实现自定义 EventBus**，而是使用 LangGraph 内置的事件处理机制。

## LangGraph 内置事件机制

### 1. CallbackHandler (同步回调)

用于日志记录、指标收集、追踪。

**现有实现:** `app/agent/callbacks/handler.py`

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

**支持的事件:**
- `on_llm_start` / `on_llm_end` / `on_llm_error`
- `on_chat_model_start` / `on_chat_model_end`
- `on_tool_start` / `on_tool_end` / `on_tool_error`
- `on_chain_start` / `on_chain_end` / `on_chain_error`

### 2. astream_events (流式事件)

用于实时流式输出和监控。

**现有实现:** `app/agent/streaming/streaming.py`

```python
async for event in agent.astream_events(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config=config,
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="", flush=True)
```

**事件类型:**
| 事件 | 说明 |
|------|------|
| `on_chat_model_start` | 聊天模型开始 |
| `on_chat_model_stream` | 聊天模型流式输出 |
| `on_chat_model_end` | 聊天模型结束 |
| `on_tool_start` | 工具调用开始 |
| `on_tool_end` | 工具调用结束 |
| `on_chain_start` | 链开始 |
| `on_chain_end` | 链结束 |

### 3. FastAPI SSE 流式响应

**现有实现:** `stream_agent_sse()`

```python
from app.agent.streaming.streaming import stream_agent_sse
from fastapi.responses import StreamingResponse

async def generate():
    async for sse in stream_agent_sse(agent, input_data):
        yield sse

return StreamingResponse(generate(), media_type="text/event-stream")
```

## 与 WeKnora99 EventBus 的对应关系

| WeKnora99 EventType | LangGraph 对应机制 |
|---------------------|-------------------|
| `query.received` | `on_chain_start` |
| `retrieval.start` | `on_tool_start` (knowledge_search) |
| `retrieval.end` | `on_tool_end` (knowledge_search) |
| `llm.start` | `on_chat_model_start` |
| `llm.end` | `on_chat_model_end` |
| `llm.stream` | `on_chat_model_stream` |
| `tool.start` | `on_tool_start` |
| `tool.end` | `on_tool_end` |
| `error` | `on_llm_error` / `on_tool_error` |

## 自定义事件处理

如果需要自定义事件处理，扩展 `KikiCallbackHandler`:

```python
from app.agent.callbacks.handler import KikiCallbackHandler

class CustomCallbackHandler(KikiCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        super().on_chat_model_start(serialized, messages, **kwargs)
        # 自定义逻辑
        print(f"Custom: Starting chat with {len(messages)} messages")
```

## 与 ChatPipeline 集成

`ChatPipeline` 通过 `RunnableConfig` 传递回调:

```python
from app.agent.callbacks import KikiCallbackHandler

handler = KikiCallbackHandler(session_id=session_id)
config = {"callbacks": [handler]}

# 在 pipeline 中使用
response = await llm_service.chat(
    messages=messages,
    config=config,  # 传递回调配置
)
```

## 参考

- [LangGraph Streaming](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming-events)
- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/)
