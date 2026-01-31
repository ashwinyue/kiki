# LangGraph 限流集成

## 概述

LangChain 和 LangGraph **本身没有内置限流功能**，但提供了多种集成方式实现限流机制。

## 实现方案

### 1. LangChain 层面 - 自定义回调处理器

使用 Token 计数回调来跟踪和控制使用量：

```python
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
import tiktoken

class RateLimitCallback(BaseCallbackHandler):
    """Token 计数和速率限制回调"""

    def __init__(self, tokens_per_minute: int = 10000):
        self.tokens_per_minute = tokens_per_minute
        self.token_count = 0
        self.start_time = None
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """在 LLM 调用前检查限流"""
        for prompt in prompts:
            tokens = self._count_tokens(prompt)
            self.token_count += tokens

            # 这里可以触发限流逻辑
            if self._should_limit():
                raise RateLimitError(f"Token limit exceeded: {self.token_count}")

    def on_llm_end(self, result: LLMResult, **kwargs) -> None:
        """记录输出 tokens"""
        for generation in result.generations:
            for gen in generation:
                self.token_count += self._count_tokens(gen.text)

    def _should_limit(self) -> bool:
        # 实现你的限流逻辑
        return self.token_count > self.tokens_per_minute

# 使用
llm = ChatOpenAI(model="gpt-4o", callbacks=[RateLimitCallback()])
```

### 2. LangGraph 层面 - 状态级限流

在 State 中集成限流计数器：

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class RateLimitedState(TypedDict):
    messages: Annotated[list, add_messages]
    token_count: int  # 已使用 token 数
    request_count: int  # 请求计数
    last_request_time: float  # 上次请求时间

def rate_limit_check(state: RateLimitedState) -> dict:
    """节点前检查限流"""
    import time

    max_tokens_per_minute = 10000
    max_requests_per_minute = 60
    window_seconds = 60

    current_time = time.time()

    # 重置窗口
    if state["last_request_time"] and (current_time - state["last_request_time"]) > window_seconds:
        return {
            "token_count": 0,
            "request_count": 0,
            "last_request_time": current_time
        }

    # 检查限制
    if state["token_count"] >= max_tokens_per_minute:
        return {
            "messages": [("assistant", "抱歉，已达到每分钟 token 限制")]
        }

    if state["request_count"] >= max_requests_per_minute:
        return {
            "messages": [("assistant", "抱歉，请求过于频繁")]
        }

    return {}

def agent_node(state: RateLimitedState) -> dict:
    """Agent 节点 - 更新计数器"""
    import time

    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(state["messages"])

    # 简单估算 token（实际应使用 tiktoken）
    estimated_tokens = len(str(response.content)) // 4

    return {
        "messages": [response],
        "token_count": state["token_count"] + estimated_tokens,
        "request_count": state["request_count"] + 1,
        "last_request_time": time.time()
    }

# 构建带限流的图
graph = StateGraph(RateLimitedState)
graph.add_node("rate_limit_check", rate_limit_check)
graph.add_node("agent", agent_node)

graph.add_edge(START, "rate_limit_check")
graph.add_conditional_edges(
    "rate_limit_check",
    lambda s: END if s.get("limit_reached") else "agent",
    {"agent": "agent", END: END}
)
graph.add_edge("agent", END)
```

### 3. 应用层面 - 使用 SlowAPI（推荐用于 FastAPI）

在 Kiki 项目中，使用中间件实现限流：

```python
# app/middleware/rate_limit.py
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from functools import wraps

limiter = Limiter(key_func=get_remote_address)

# 在路由中使用
from fastapi import APIRouter

router = APIRouter()

@router.post("/chat")
@limiter.limit("60/minute")  # 每分钟 60 次
async def chat_endpoint(request: Request, message: str):
    # 你的聊天逻辑
    pass
```

### 4. LLM Provider 层面 - 使用 Tenacity 重试

结合指数退避处理 API 限流：

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError
from langchain_openai import ChatOpenAI

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError)
)
def call_llm_with_retry(messages):
    llm = ChatOpenAI(model="gpt-4o")
    return llm.invoke(messages)
```

## Kiki 项目推荐方案

基于 FastAPI + LangGraph 架构：

| 层级 | 方案 | 适用场景 |
|-----|------|---------|
| **API 层** | `slowapi` 中间件 | 按 IP/用户限流请求 |
| **应用层** | Redis 计数器 | 全局限流，多实例共享 |
| **LangGraph** | State 计数器 | 会话级 token 限流 |
| **LLM 层** | Tenacity 重试 | 处理 Provider 限流 |

## 依赖安装

```bash
# 限流相关
uv add slowapi
uv add tenacity
uv add tiktoken

# Redis（可选，用于分布式限流）
uv add redis
```

## 参考

- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/)
- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [SlowAPI Documentation](https://slowapi.readthedocs.io/)
