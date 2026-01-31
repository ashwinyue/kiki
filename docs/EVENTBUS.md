# EventBus 集成评估报告

> 版本: v0.1.0
> 更新日期: 2025-01-31
> 参考来源: WeKnora 重构规划、LangGraph 最佳实践

## 目录

- [概述](#概述)
- [WeKnora EventBus 设计方案](#weknora-eventbus-设计方案)
- [Kiki 现有事件机制](#kiki-现有事件机制)
- [LangGraph/LangChain 最佳实践](#langgraphlangchain-最佳实践)
- [集成建议](#集成建议)
- [决策矩阵](#决策矩阵)

---

## 概述

本文档评估 Kiki 企业级 Agent 开发脚手架是否需要引入独立的 EventBus 组件，基于 WeKnora 项目的 EventBus 设计方案和 LangGraph/LangChain 的最佳实践进行分析。

### 核心结论

| 场景 | 是否需要 EventBus | 推荐方案 |
|------|------------------|----------|
| 单 Agent 应用 | ❌ 不需要 | 使用现有 StreamEvent + CallbackHandler |
| Multi-Agent 单节点协作 | ❌ 不需要 | 使用 LangGraph 原生图编排 |
| Multi-Agent 跨节点协作 | ✅ 需要 | 阶段1: Redis Pub/Sub，阶段2: Apache Pulsar |
| 异步任务解耦 | ✅ 需要 | 轻量级 EventBus（基于 Redis） |
| 事件溯源/审计 | ✅ 需要 | 持久化事件存储 + EventBus |

---

## WeKnora EventBus 设计方案

WeKnora 基于 Go/Eino 的重构规划中设计了完整的事件总线架构：

### 技术选型

| 维度 | 方案 |
|------|------|
| **实现基础** | Asynq（基于 Redis 的任务队列） |
| **消息持久化** | Redis |
| **升级路径** | 阶段1: Asynq → 阶段2: Apache Pulsar |

### 事件类型定义

```go
// Agent 生命周期事件
const (
    TypeAgentStarted    = "agent:started"
    TypeAgentCompleted  = "agent:completed"
    TypeAgentFailed     = "agent:failed"
    TypeAgentMessage    = "agent:message"
    TypeMultiAgentSync  = "multiagent:sync"
    TypeAgentToolCall   = "agent:tool_call"
    TypeAgentToolResult = "agent:tool_result"
)
```

### 核心设计模式

#### 1. 事件发布（通过中间件）

```go
// Eino Agent 中间件 - 发布事件
func EventPublishingMiddleware(eventBus EventBus) AgentMiddleware {
    return AgentMiddleware{
        Before: func(ctx context.Context, msg *schema.Message) error {
            agentID, _ := ctx.Value("agentID").(string)
            return eventBus.PublishAgentEvent(ctx, agentID, TypeAgentStarted, map[string]interface{}{
                "message_role":    msg.Role,
                "message_content": msg.Content,
            })
        },
        After: func(ctx context.Context, msg *schema.Message) error {
            agentID, _ := ctx.Value("agentID").(string)
            return eventBus.PublishAgentEvent(ctx, agentID, TypeAgentCompleted, map[string]interface{}{
                "message_role":    msg.Role,
                "message_content": msg.Content,
            })
        },
    }
}
```

#### 2. 事件订阅

```go
func HandleAgentStartedEvent(ctx context.Context, t *asynq.Task) error {
    var payload AgentEventPayload
    if err := json.Unmarshal(t.Payload(), &payload); err != nil {
        return fmt.Errorf("failed to unmarshal event: %w", err)
    }

    logger.Infof(ctx, "Agent started: agent_id=%s, event_id=%s",
        payload.AgentID, payload.EventID)

    // 更新 Agent 状态、发送通知、触发其他业务逻辑
    return nil
}
```

#### 3. Multi-Agent 事件协调

```go
type MultiAgentCoordinator struct {
    eventBus EventBus
    agents   map[string]Agent
}

func (c *MultiAgentCoordinator) SubscribeAgentEvents(ctx context.Context, agentID string) error {
    return c.eventBus.Subscribe(ctx, TypeMultiAgentSync, func(ctx context.Context, t *asynq.Task) error {
        var payload AgentEventPayload
        json.Unmarshal(t.Payload(), &payload)

        if payload.AgentID != agentID {
            return nil
        }

        if agent, exists := c.agents[agentID]; exists {
            return agent.HandleEvent(ctx, &payload)
        }
        return nil
    })
}
```

### 对比 Coze-Studio Pulsar 方案

| 特性 | Coze-Studio (Pulsar) | WeKnora (Asynq) | 优势 |
|------|---------------------|-----------------|------|
| 消息持久化 | ✅ 强一致性 | ✅ 基于Redis | 都支持 |
| 优先级队列 | ✅ 支持 | ✅ 支持 | 都支持 |
| 消息重试 | ✅ 支持 | ✅ 支持 | 都支持 |
| 部署复杂度 | 高（需要Pulsar集群） | 低（基于Redis） | Asynq更优 |
| 吞吐量 | 极高 | 高 | Pulsar更优 |
| 消息路由 | 复杂（支持分区、订阅模式） | 简单（基于任务类型） | Pulsar更灵活 |
| 学习曲线 | 陡峭 | 平缓 | Asynq更易用 |
| 运维成本 | 高 | 低 | Asynq更优 |

---

## Kiki 现有事件机制

Kiki 已有成熟的事件处理基础设施，无需引入额外组件即可满足大部分场景：

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        Kiki 事件系统                         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LangGraph astream_events               │   │
│  │  - on_chat_model_start/end                          │   │
│  │  - on_llm_start/end                                 │   │
│  │  - on_tool_start/end                                │   │
│  │  - on_chain_start/end                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              StreamEvent 体系                        │   │
│  │  - TokenEvent      (Token 流式输出)                  │   │
│  │  - ToolCallEvent   (工具调用事件)                    │   │
│  │  - ErrorEvent      (错误事件)                        │   │
│  │  - StatusEvent     (状态事件)                        │   │
│  │  - DoneEvent       (完成事件)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              KikiCallbackHandler                     │   │
│  │  - Langfuse 追踪                                     │   │
│  │  - Prometheus 指标                                   │   │
│  │  - structlog 日志                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SSE 实时推送                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 现有能力清单

| 能力 | 实现位置 | 说明 |
|------|----------|------|
| **流式事件** | `app/agent/streaming.py` | StreamEvent 体系 + SSE 输出 |
| **生命周期追踪** | `app/agent/callbacks/handler.py` | KikiCallbackHandler |
| **指标收集** | `app/agent/callbacks/metrics.py` | MetricsCallbackHandler |
| **状态持久化** | `app/core/store.py` | CheckpointSaver |
| **Redis 基础设施** | `app/infra/redis.py` | 可扩展为 Pub/Sub |
| **可观测性** | `app/observability/` | Langfuse + Prometheus + structlog |

### 事件类型对比

| WeKnora Event | Kiki 现有对应 | 覆盖情况 |
|--------------|---------------|----------|
| `agent:started` | `on_chat_model_start` + `StatusEvent("start")` | ✅ 已覆盖 |
| `agent:completed` | `on_chat_model_end` + `DoneEvent` | ✅ 已覆盖 |
| `agent:failed` | `on_llm_error` + `ErrorEvent` | ✅ 已覆盖 |
| `agent:tool_call` | `on_tool_start` + `ToolCallEvent` | ✅ 已覆盖 |
| `agent:tool_result` | `on_tool_end` + `ToolCallEvent` | ✅ 已覆盖 |
| `multiagent:sync` | 无（跨节点通信） | ❌ 未覆盖 |

---

## LangGraph/LangChain 最佳实践

### 官方推荐的事件处理模式

| 实践 | 说明 |
|------|------|
| **优先使用 `astream_events()`** | LangGraph 内置的事件流 API，支持细粒度事件监听 |
| **Callback 用于可观测性** | 不用于业务逻辑，仅用于追踪/监控 |
| **Redis 用于持久化/缓存** | CheckpointSaver 实现状态持久化 |
| **WebSocket 用于实时通信** | 客户端实时通知优先选择 WebSocket 而非 SSE |
| **避免过度设计** | 单 Agent 场景不需要复杂的事件总线 |

### astream_events() 使用示例

```python
from langgraph.types import RunnableConfig

async for event in graph.astream_events(
    input_data,
    config,
    version="v1",
):
    event_type = event["event"]
    data = event["data"]

    if event_type == "on_chat_model_start":
        print(f"模型开始调用: {data['input']}")

    elif event_type == "on_chat_model_stream":
        print(f"流式输出: {data['chunk'].content}")

    elif event_type == "on_tool_start":
        print(f"工具调用: {data['input']['name']}")

    elif event_type == "on_tool_end":
        print(f"工具结果: {data['output']}")
```

### 内置事件类型

```python
# LangChain Core Events
"on_llm_start"
"on_llm_end"
"on_llm_error"
"on_chat_model_start"
"on_chat_model_end"
"on_chat_model_stream"
"on_chain_start"
"on_chain_end"
"on_chain_error"
"on_tool_start"
"on_tool_end"
"on_tool_error"
"on_retriever_start"
"on_retriever_end"
"on_retriever_error"
```

---

## 集成建议

### 阶段 1：基于现有能力扩展（推荐优先）

**暂不引入独立 EventBus**，充分利用 LangGraph 原生能力。

#### 适用场景
- ✅ 单 Agent 应用
- ✅ 需要实时流式输出
- ✅ 需要可观测性（Langfuse/Prometheus）
- ✅ Multi-Agent 单节点协作

#### 扩展方案

```python
# 1. 扩展现有 StreamEvent 体系
# app/agent/streaming.py

class AgentLifecycleEvent(StreamEvent):
    """Agent 生命周期事件（新增）"""

    def __init__(
        self,
        agent_id: str,
        event_type: str,  # started, completed, failed
        data: dict | None = None,
    ):
        super().__init__(
            "agent_lifecycle",
            {
                "agent_id": agent_id,
                "event_type": event_type,
                "data": data or {},
            },
        )

# 2. 在现有 KikiCallbackHandler 中发布事件
# app/agent/callbacks/handler.py

class KikiCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        enable_event_publish: bool = False,  # 新增
    ):
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self.enable_event_publish = enable_event_publish

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # 现有追踪逻辑...
        logger.info("chat_model_start", session_id=self.session_id)

        # 新增：可选发布事件到 Redis
        if self.enable_event_publish:
            await self._publish_event("agent:started", {
                "agent_id": kwargs.get("agent_id"),
                "model": serialized.get("name"),
            })

    async def _publish_event(self, event_type: str, data: dict):
        """可选：通过 Redis Pub/Sub 发布跨节点事件"""
        from app.infra.redis import get_redis

        redis = await get_redis()
        await redis.publish(f"agent:{event_type}", json.dumps(data))
```

### 阶段 2：引入轻量级 EventBus（按需）

**触发条件**：满足以下任一需求时再引入

1. **Multi-Agent 跨节点协作** - 不同服务实例间的 Agent 需要通信
2. **异步任务解耦** - Agent 执行后需要触发后台任务
3. **事件溯源** - 需要完整的事件历史记录用于审计/重放

#### 推荐实现：基于 Redis Pub/Sub

```python
# app/core/eventbus.py
from typing import Callable, Awaitable
import json
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AgentEventType:
    """Agent 事件类型（参考 WeKnora）"""
    STARTED = "agent:started"
    COMPLETED = "agent:completed"
    FAILED = "agent:failed"
    TOOL_CALL = "agent:tool_call"
    TOOL_RESULT = "agent:tool_result"
    MULTI_AGENT_SYNC = "multiagent:sync"


class EventBus:
    """轻量级事件总线（基于 Redis Pub/Sub）"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self._subscribers: dict[str, list[Callable]] = {}
        self._listener_task: asyncio.Task | None = None

    async def publish(self, topic: str, event: dict) -> None:
        """发布事件"""
        channel = f"agent:{topic}"
        payload = json.dumps(event, ensure_ascii=False)
        await self.redis.publish(channel, payload)
        logger.debug("event_published", topic=topic, event=event)

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """订阅事件（进程内）"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)

    async def start_listener(self) -> None:
        """启动跨节点事件监听"""
        if self._listener_task:
            return

        async def _listener():
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("agent:*")

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    await self._handle_message(message)

        self._listener_task = asyncio.create_task(_listener())

    async def _handle_message(self, message: dict) -> None:
        """处理接收到的消息"""
        try:
            channel = message['channel'].decode()
            data = json.loads(message['data'])

            # 提取 topic
            topic = channel.replace("agent:", "")

            # 调用订阅者
            for handler in self._subscribers.get(topic, []):
                await handler(data)

        except Exception as e:
            logger.error("event_handle_failed", error=str(e))

    async def close(self) -> None:
        """关闭事件总线"""
        if self._listener_task:
            self._listener_task.cancel()
            self._listener_task = None


# 全局单例
_eventbus: EventBus | None = None


async def get_eventbus() -> EventBus:
    """获取事件总线单例"""
    global _eventbus
    if _eventbus is None:
        from app.infra.redis import get_redis
        redis = await get_redis()
        _eventbus = EventBus(redis)
        await _eventbus.start_listener()
    return _eventbus
```

#### 使用示例

```python
# 发布事件
from app.core.eventbus import get_eventbus, AgentEventType

eventbus = await get_eventbus()
await eventbus.publish(
    AgentEventType.STARTED,
    {
        "agent_id": "agent-123",
        "user_id": "user-456",
        "session_id": "session-789",
        "timestamp": "2025-01-31T10:00:00Z",
    }
)

# 订阅事件
async def handle_agent_started(event: dict):
    agent_id = event.get("agent_id")
    logger.info("agent_started_handler", agent_id=agent_id)
    # 处理业务逻辑...

await eventbus.subscribe(AgentEventType.STARTED, handle_agent_started)
```

### 阶段 3：企业级升级（未来）

当以下情况出现时，考虑升级到 Apache Pulsar：

| 指标 | 触发阈值 |
|------|----------|
| 消息吞吐量 | > 10,000 msg/s |
| 事件类型数量 | > 50 |
| 订阅者数量 | > 20 |
| 跨数据中心部署 | 是 |
| 消息留存要求 | > 7 天 |

---

## 决策矩阵

```
                    ┌─────────────────────────────────────┐
                    │         是否需要 EventBus？          │
                    └─────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │   单 Agent 应用        │         │   Multi-Agent 系统    │
        │   简单流程             │         │   跨节点协作          │
        └───────────────────────┘         └───────────────────────┘
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ ❌ 不需要 EventBus     │         │ ✅ 需要 EventBus       │
        │ 使用现有:              │         │ 阶段1: Redis Pub/Sub   │
        │ - StreamEvent         │         │ 阶段2: Apache Pulsar   │
        │ - CallbackHandler     │         │       (按需升级)        │
        │ - astream_events      │         │                        │
        └───────────────────────┘         └───────────────────────┘
```

### 不需要 EventBus 的场景

| 场景 | 原因 | 替代方案 |
|------|------|----------|
| 纯粹的日志收集 | Callback + Langfuse 已足够 | `KikiCallbackHandler` |
| 单进程内的组件通信 | 直接函数调用更简单 | `asyncio.Queue` 或直接调用 |
| 简单的 Agent 状态管理 | LangGraph Checkpoint 已支持 | `CheckpointSaver` |
| 低并发场景 | 引入 EventBus 增加复杂度 | 现有事件机制 |
| 实时流式输出 | SSE 已足够支持 | `stream_agent_sse()` |

---

## 配置示例

```yaml
# config/eventbus.yaml
eventbus:
  # 是否启用事件总线
  enabled: false

  # 事件总线提供者
  provider: redis  # redis | pulsar | kafka

  # Redis 配置
  redis:
    addr: ${REDIS_ADDR}
    password: ${REDIS_PASSWORD}
    db: 2
    channel_prefix: "agent:"

  # 事件发布配置
  publish:
    # 是否启用事件发布
    enabled: false
    # 异步发布（非阻塞）
    async: true
    # 发布超时
    timeout: 5s

  # 事件订阅配置
  subscribe:
    # 订阅的事件类型
    events:
      - agent:started
      - agent:completed
      - agent:failed
      - agent:tool_call
      - multiagent:sync
    # 工作线程数
    workers: 4

  # 持久化配置
  persistence:
    # 是否持久化事件
    enabled: false
    # 持久化存储
    storage: postgres  # postgres | redis | filesystem
    # 保留天数
    retention_days: 7
```

---

## 参考资源

### 内部文档
- [Agent 系统文档](AGENT.md)
- [架构文档](ARCHITECTURE.md)
- [开发文档](DEVELOPMENT.md)

### 外部资源
- [LangGraph Streaming - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Building Event-Driven Multi-Agent Workflows](https://medium.com/@_Ankit_Malviya/building-event-driven-multi-agent-workflows-with-triggers-in-langgraph-48386c0aac5d)
- [Event-Driven Patterns for AI Agents - Reddit](https://www.reddit.com/r/LangChain/comments/1ha8mrc/eventdriven_patterns_for_ai_agents/)
- [LangGraph & Redis: Build Smarter AI Agents](https://pub.towardsai.net/langgraph-redis-build-smarter-ai-agents-with-memory-persistence-49d81a66ac61)
- [Building Production Agentic AI Systems in 2026](https://brlikhon.engineer/blog/building-production-agentic-ai-systems-in-2026-langgraph-vs-autogen-vs-crewai-complete-architecture-guide)

### WeKnora 参考
- `aold/miniblog/docs/weknora-refactor-plan.md` - WeKnora 重构规划文档
