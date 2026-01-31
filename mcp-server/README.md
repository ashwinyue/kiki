# Kiki MCP Server

Kiki Agent Framework 的 Model Context Protocol 服务器。

## 快速开始

### 使用 uv 启动 (推荐)

```bash
uv --directory /path/to/kiki/mcp-server run run_server.py
```

### 使用 Python 启动

```bash
cd /path/to/kiki/mcp-server
pip install -r requirements.txt
python run_server.py
```

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `KIKI_BASE_URL` | Kiki API 地址 | `http://localhost:8000/api/v1` |
| `KIKI_API_KEY` | Kiki API 密钥 | - |
| `KIKI_TIMEOUT` | 请求超时时间(秒) | `120` |

## MCP 客户端配置

详细配置请参考 [MCP_CONFIG.md](./MCP_CONFIG.md)

```json
{
  "mcpServers": {
    "kiki": {
      "command": "uv",
      "args": ["--directory", "/path/to/kiki/mcp-server", "run", "run_server.py"],
      "env": {
        "KIKI_API_KEY": "kiki_xxxxx",
        "KIKI_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

## 可用工具

### Agent 管理
- `list_agents` - 列出所有 Agent
- `get_agent` - 获取 Agent 详情
- `get_agent_stats` - 获取 Agent 统计
- `create_agent` - 创建 Agent
- `update_agent` - 更新 Agent
- `delete_agent` - 删除 Agent

### 聊天与会话
- `chat` - 发送聊天消息
- `get_chat_history` - 获取聊天历史
- `clear_chat_history` - 清除聊天历史
- `get_context_stats` - 获取上下文统计

### 工具
- `list_available_tools` - 列出可用工具

### 多 Agent 系统
- `list_agent_systems` - 列出多 Agent 系统
- `get_agent_system` - 获取多 Agent 系统详情
- `delete_agent_system` - 删除多 Agent 系统

### 执行历史
- `list_executions` - 列出执行历史

## License

MIT
