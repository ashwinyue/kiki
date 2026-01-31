# Kiki MCP Server 配置指南

> 更推荐使用 `uv` 来运行基于 Python 的 MCP 服务。

## 1. 安装 uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew (macOS)
brew install uv

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. MCP 客户端配置

### Claude Desktop 配置

在 Claude Desktop 设置中添加:

```json
{
  "mcpServers": {
    "kiki": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/kiki/mcp-server",
        "run",
        "run_server.py"
      ],
      "env": {
        "KIKI_API_KEY": "kiki_xxxxx",
        "KIKI_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

### Cursor 配置

在 Cursor 中，编辑 MCP 配置文件 (通常在 `~/.cursor/mcp-config.json`):

```json
{
  "mcpServers": {
    "kiki": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/kiki/mcp-server",
        "run",
        "run_server.py"
      ],
      "env": {
        "KIKI_API_KEY": "kiki_xxxxx",
        "KIKI_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

### Cline (Claude Code) 配置

编辑 `~/.claude/mcp_config.json`:

```json
{
  "mcpServers": {
    "kiki": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/kiki/mcp-server",
        "run",
        "run_server.py"
      ],
      "env": {
        "KIKI_API_KEY": "kiki_xxxxx",
        "KIKI_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

### 其他 MCP 客户端

```json
{
  "mcpServers": {
    "kiki": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/kiki/mcp-server",
        "run",
        "run_server.py"
      ],
      "env": {
        "KIKI_API_KEY": "kiki_xxxxx",
        "KIKI_BASE_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

## 3. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `KIKI_BASE_URL` | Kiki API 地址 | `http://localhost:8000/api/v1` |
| `KIKI_API_KEY` | Kiki API 密钥 | - |
| `KIKI_TIMEOUT` | 请求超时时间(秒) | `120` |

## 4. 获取 API Key

1. 启动 Kiki 服务
2. 访问 `http://localhost:8000/docs`
3. 使用 `/api/v1/auth/login` 端点登录获取 API Key
4. API Key 格式: `kiki_xxxxx`

## 5. 可用工具

| 工具名 | 说明 |
|--------|------|
| `list_agents` | 列出所有 Agent |
| `get_agent` | 获取 Agent 详情 |
| `create_agent` | 创建新 Agent |
| `update_agent` | 更新 Agent |
| `delete_agent` | 删除 Agent |
| `chat` | 发送聊天消息 |
| `get_chat_history` | 获取聊天历史 |
| `clear_chat_history` | 清除聊天历史 |
| `list_available_tools` | 列出可用工具 |
| `list_agent_systems` | 列出多 Agent 系统 |
