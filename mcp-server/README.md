# Kiki MCP Server

A Model Context Protocol (MCP) server that provides access to the Kiki Agent Framework API.

## Features

This MCP server exposes the following Kiki functionality through the MCP protocol:

### Agent Management
- `list_agents` - List all agents with optional filtering
- `get_agent` - Get detailed information about a specific agent
- `get_agent_stats` - Get statistics about all agents
- `create_agent` - Create a new agent
- `update_agent` - Update an existing agent
- `delete_agent` - Delete an agent

### Chat & Session
- `chat` - Send a message to an agent and get a response
- `get_chat_history` - Get chat history for a session
- `clear_chat_history` - Clear chat history for a session
- `get_context_stats` - Get context statistics for a session

### Tools
- `list_available_tools` - List all available tools

### Multi-Agent Systems
- `list_agent_systems` - List all multi-agent systems
- `get_agent_system` - Get details of a multi-agent system
- `delete_agent_system` - Delete a multi-agent system

### Executions
- `list_executions` - List agent execution history

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Linux/macOS
export KIKI_BASE_URL="http://localhost:8000/api/v1"
export KIKI_API_KEY="your_api_key_here"  # Optional

# Windows PowerShell
$env:KIKI_BASE_URL="http://localhost:8000/api/v1"
$env:KIKI_API_KEY="your_api_key_here"  # Optional
```

### 3. Run the Server

```bash
python main.py
```

### 4. Command Line Options

```bash
python main.py --help                 # Show help
python main.py --check-only           # Check environment only
python main.py --verbose              # Enable verbose logging
python main.py --version              # Show version
```

## Installation as Python Package

### Development Mode

```bash
pip install -e .
```

After installation, you can run:

```bash
kiki-mcp-server
```

### Production Mode

```bash
pip install .
```

## MCP Configuration

To use this MCP server with Claude Desktop or other MCP clients, add the following to your MCP config:

```json
{
  "mcpServers": {
    "kiki": {
      "command": "python",
      "args": ["/path/to/kiki/mcp-server/main.py"],
      "env": {
        "KIKI_BASE_URL": "http://localhost:8000/api/v1",
        "KIKI_API_KEY": "kiki_mcp_your_api_key_here"
      }
    }
  }
}
```

### Quick Setup (One-time)

1. **Start Kiki server**:
   ```bash
   cd /path/to/kiki
   uv run uvicorn app.main:app --reload
   ```

2. **Create a user and login**:
   ```bash
   # Register (if needed)
   curl -X POST "http://localhost:8000/api/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "email": "admin@example.com", "password": "secure_password"}'

   # Login
   curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "secure_password"}'
   ```

3. **Create MCP API Key**:
   ```bash
   # Use the JWT token from login response
   export TOKEN="your_jwt_token_here"

   curl -X POST "http://localhost:8000/api/v1/api-keys/mcp/create?name=Claude%20Desktop" \
     -H "Authorization: Bearer $TOKEN"
   ```

4. **Copy the API Key** from the response and use it in your MCP config.

## Usage Examples

### Listing Agents

```python
# Call the list_agents tool
{
  "name": "list_agents",
  "arguments": {
    "agent_type": "chat",
    "status": "active",
    "page": 1,
    "size": 10
  }
}
```

### Creating an Agent

```python
# Call the create_agent tool
{
  "name": "create_agent",
  "arguments": {
    "name": "My Assistant",
    "description": "A helpful assistant",
    "agent_type": "chat",
    "model_name": "gpt-4o",
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7
  }
}
```

### Chatting with an Agent

```python
# Call the chat tool
{
  "name": "chat",
  "arguments": {
    "message": "Hello, how are you?",
    "session_id": "my-session-123"
  }
}
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `KIKI_BASE_URL` | Kiki API base URL | `http://localhost:8000/api/v1` |
| `KIKI_API_KEY` | API key or JWT token for authentication | (empty) |
| `KIKI_TIMEOUT` | Request timeout in seconds | `120` |

### Authentication

The MCP server supports two authentication methods:

1. **Kiki API Key** (recommended for MCP): Format: `kiki_xxxx`
   - Create via Kiki API: `POST /api/v1/api-keys/mcp/create`
   - Has scoped permissions (chat, agents:read, tools:read)

2. **JWT Bearer Token**: Format: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - Get from login: `POST /api/v1/auth/login`
   - Full user permissions

### Creating an MCP API Key

To create a dedicated API Key for MCP server usage:

```bash
# First, login to get a JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Then create an MCP API Key
curl -X POST "http://localhost:8000/api/v1/api-keys/mcp/create?name=My%20MCP%20Server" \
  -H "Authorization: Bearer <your_jwt_token>"

# Response will contain the API Key (save it, it won't be shown again!)
# {
#   "id": 1,
#   "key": "kiki_mcp_abc123...",
#   "key_prefix": "kiki_mcp",
#   ...
# }
```

## Troubleshooting

### Import Errors

If you encounter import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version compatibility (3.10+)
3. Verify no file name conflicts (avoid naming files `mcp.py`)

### Connection Errors

If you can't connect to the Kiki API:
1. Verify `KIKI_BASE_URL` is correct
2. Check that the Kiki server is running
3. Verify network connectivity
4. Check if authentication is required

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
ruff check .
```

## License

MIT License

## Support

For issues and questions, please visit: https://github.com/your-org/kiki/issues
