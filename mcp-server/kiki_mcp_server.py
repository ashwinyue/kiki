#!/usr/bin/env python3
"""Kiki MCP Server

A Model Context Protocol server that provides access to the Kiki Agent Framework API.

This server exposes Kiki's Agent management, session management, and chat functionality
through the MCP protocol, allowing AI assistants to interact with Kiki agents.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict

import httpx
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
KIKI_BASE_URL = os.getenv("KIKI_BASE_URL", "http://localhost:8000/api/v1")
# API Key for authentication (format: kiki_xxxx or Bearer token)
KIKI_API_KEY = os.getenv("KIKI_API_KEY", "")
KIKI_TIMEOUT = float(os.getenv("KIKI_TIMEOUT", "120"))

# Determine auth header format
# If API key starts with "kiki_", send it directly
# Otherwise, use Bearer format (for JWT tokens)
if KIKI_API_KEY and not KIKI_API_KEY.startswith("kiki_"):
    # JWT Token or other Bearer token
    AUTH_HEADER_FORMAT = "bearer"
else:
    # Kiki API Key (kiki_xxxx format)
    AUTH_HEADER_FORMAT = "api_key"


class KikiClient:
    """HTTP client for interacting with Kiki API"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        auth_format: str = "api_key",
    ):
        """Initialize the Kiki API client

        Args:
            base_url: Base URL of the Kiki API
            api_key: API key or JWT token for authentication
            timeout: Request timeout in seconds
            auth_format: Auth format ("api_key" or "bearer")
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._auth_format = auth_format
        self._client: httpx.AsyncClient | None = None
        self._headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            if auth_format == "bearer":
                # JWT Token format
                self._headers["Authorization"] = f"Bearer {api_key}"
            else:
                # Kiki API Key format (kiki_xxxx)
                # Can be sent as Authorization header with Bearer prefix
                # or as X-API-Key header
                self._headers["Authorization"] = f"Bearer {api_key}"
                self._headers["X-API-Key"] = api_key

    async def __aenter__(self):
        """Async context manager entry"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Kiki API

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx

        Returns:
            JSON response as dictionary

        Raises:
            httpx.HTTPError: If the request fails
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"/{endpoint.lstrip('/')}"
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed: {e.response.status_code} - {e.response.text}")
            # Try to parse error response
            try:
                error_data = e.response.json()
                raise RuntimeError(error_data.get("detail", str(e))) from e
            except Exception:
                raise RuntimeError(f"API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise RuntimeError(f"Request error: {str(e)}") from e

    # ==================== Agent Management ====================

    async def list_agents(
        self,
        agent_type: str | None = None,
        status: str | None = None,
        page: int = 1,
        size: int = 20
    ) -> Dict[str, Any]:
        """List all agents with optional filtering

        Args:
            agent_type: Filter by agent type
            status: Filter by agent status
            page: Page number
            size: Page size

        Returns:
            List of agents with pagination info
        """
        params = {"page": page, "size": size}
        if agent_type:
            params["agent_type"] = agent_type
        if status:
            params["status"] = status
        return await self._request("GET", "agents/list", params=params)

    async def get_agent(self, agent_id: int) -> Dict[str, Any]:
        """Get agent details by ID

        Args:
            agent_id: Agent ID

        Returns:
            Agent details
        """
        return await self._request("GET", f"agents/{agent_id}")

    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics

        Returns:
            Agent statistics
        """
        return await self._request("GET", "agents/stats")

    async def create_agent(
        self,
        name: str,
        agent_type: str,
        model_name: str,
        description: str = "",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        config: Dict[str, Any] | None = None,
        tool_ids: list[int] | None = None,
    ) -> Dict[str, Any]:
        """Create a new agent

        Args:
            name: Agent name
            agent_type: Agent type (chat, react, router, supervisor, etc.)
            model_name: Model name to use
            description: Agent description
            system_prompt: System prompt for the agent
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            config: Additional configuration
            tool_ids: List of tool IDs to attach

        Returns:
            Created agent details
        """
        data = {
            "name": name,
            "description": description,
            "agent_type": agent_type,
            "model_name": model_name,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "config": config or {},
            "tool_ids": tool_ids or [],
        }
        return await self._request("POST", "agents", json=data)

    async def update_agent(
        self,
        agent_id: int,
        name: str | None = None,
        description: str | None = None,
        agent_type: str | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config: Dict[str, Any] | None = None,
        tool_ids: list[int] | None = None,
    ) -> Dict[str, Any]:
        """Update an existing agent

        Args:
            agent_id: Agent ID
            name: New agent name
            description: New description
            agent_type: New agent type
            model_name: New model name
            system_prompt: New system prompt
            temperature: New temperature
            max_tokens: New max tokens
            config: New configuration
            tool_ids: New tool IDs

        Returns:
            Updated agent details
        """
        # Get current agent data first
        current = await self.get_agent(agent_id)

        # Build update data with only provided fields
        data = {
            "name": name or current["name"],
            "description": description if description is not None else current["description"],
            "agent_type": agent_type or current["agent_type"],
            "model_name": model_name or current["model_name"],
            "system_prompt": system_prompt if system_prompt is not None else current["system_prompt"],
            "temperature": temperature if temperature is not None else current["temperature"],
            "max_tokens": max_tokens if max_tokens is not None else current["max_tokens"],
            "config": config if config is not None else current["config"],
            "tool_ids": tool_ids if tool_ids is not None else current.get("tool_ids", []),
        }
        return await self._request("PATCH", f"agents/{agent_id}", json=data)

    async def delete_agent(self, agent_id: int) -> None:
        """Delete an agent

        Args:
            agent_id: Agent ID
        """
        await self._request("DELETE", f"agents/{agent_id}")

    # ==================== Chat & Session ====================

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Send a chat message (non-streaming)

        Args:
            message: User message
            session_id: Session ID for context
            user_id: Optional user ID

        Returns:
            Chat response with content
        """
        data = {
            "message": message,
            "session_id": session_id,
        }
        if user_id:
            data["user_id"] = user_id
        return await self._request("POST", "chat", json=data)

    async def get_chat_history(self, session_id: str) -> Dict[str, Any]:
        """Get chat history for a session

        Args:
            session_id: Session ID

        Returns:
            Chat history with messages
        """
        return await self._request("GET", f"chat/history/{session_id}")

    async def clear_chat_history(self, session_id: str) -> Dict[str, str]:
        """Clear chat history for a session

        Args:
            session_id: Session ID

        Returns:
            Status response
        """
        return await self._request("DELETE", f"chat/history/{session_id}")

    async def get_context_stats(self, session_id: str) -> Dict[str, Any]:
        """Get context statistics for a session

        Args:
            session_id: Session ID

        Returns:
            Context statistics
        """
        return await self._request("GET", f"chat/context/{session_id}/stats")

    # ==================== Tools ====================

    async def list_tools(self) -> list[Dict[str, Any]]:
        """List available tools

        Returns:
            List of available tools
        """
        result = await self._request("GET", "tools")
        return result.get("tools", [])

    # ==================== Agent Systems ====================

    async def list_agent_systems(self) -> list[Dict[str, Any]]:
        """List all multi-agent systems

        Returns:
            List of agent systems
        """
        return await self._request("GET", "agents/systems")

    async def get_agent_system(self, system_id: str) -> Dict[str, Any]:
        """Get a multi-agent system

        Args:
            system_id: System ID

        Returns:
            Agent system details
        """
        return await self._request("GET", f"agents/systems/{system_id}")

    async def delete_agent_system(self, system_id: str) -> Dict[str, bool]:
        """Delete a multi-agent system

        Args:
            system_id: System ID

        Returns:
            Deletion status
        """
        return await self._request("DELETE", f"agents/systems/{system_id}")

    # ==================== Executions ====================

    async def list_executions(
        self,
        agent_id: int | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List agent execution history

        Args:
            agent_id: Filter by agent ID
            limit: Maximum number of executions

        Returns:
            List of executions
        """
        params = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        return await self._request("GET", "agents/executions", params=params)


# ==================== MCP Server ====================

app = Server("kiki-server")


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available Kiki MCP tools"""
    return [
        # Agent Management
        Tool(
            name="list_agents",
            description="List all agents with optional filtering by type and status",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_type": {
                        "type": "string",
                        "description": "Filter by agent type (chat, react, router, supervisor, swarm)",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by agent status (active, inactive, archived)",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (default: 1)",
                        "default": 1,
                    },
                    "size": {
                        "type": "integer",
                        "description": "Page size (default: 20, max: 100)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="get_agent",
            description="Get detailed information about a specific agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "integer",
                        "description": "Agent ID",
                    },
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="get_agent_stats",
            description="Get statistics about all agents",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="create_agent",
            description="Create a new agent with specified configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Agent name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Agent description",
                    },
                    "agent_type": {
                        "type": "string",
                        "description": "Agent type (chat, react, router, supervisor, swarm)",
                        "enum": ["chat", "react", "router", "supervisor", "swarm"],
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Model name (e.g., gpt-4o, claude-3-5-sonnet)",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the agent",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for generation (0.0 - 1.0)",
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": 2000,
                    },
                },
                "required": ["name", "agent_type", "model_name"],
            },
        ),
        Tool(
            name="update_agent",
            description="Update an existing agent's configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "integer",
                        "description": "Agent ID",
                    },
                    "name": {
                        "type": "string",
                        "description": "New agent name",
                    },
                    "description": {
                        "type": "string",
                        "description": "New agent description",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "New system prompt",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "New temperature",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "New max tokens",
                    },
                },
                "required": ["agent_id"],
            },
        ),
        Tool(
            name="delete_agent",
            description="Delete an agent (soft delete)",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "integer",
                        "description": "Agent ID",
                    },
                },
                "required": ["agent_id"],
            },
        ),
        # Chat & Session
        Tool(
            name="chat",
            description="Send a message to an agent and get a response",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User message to send",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation context",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user ID",
                    },
                },
                "required": ["message", "session_id"],
            },
        ),
        Tool(
            name="get_chat_history",
            description="Get chat history for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="clear_chat_history",
            description="Clear chat history for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="get_context_stats",
            description="Get context statistics for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                },
                "required": ["session_id"],
            },
        ),
        # Tools
        Tool(
            name="list_available_tools",
            description="List all available tools that can be attached to agents",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # Agent Systems
        Tool(
            name="list_agent_systems",
            description="List all multi-agent systems (Router, Supervisor, Swarm)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_agent_system",
            description="Get details of a multi-agent system",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {
                        "type": "string",
                        "description": "System ID",
                    },
                },
                "required": ["system_id"],
            },
        ),
        Tool(
            name="delete_agent_system",
            description="Delete a multi-agent system",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_id": {
                        "type": "string",
                        "description": "System ID",
                    },
                },
                "required": ["system_id"],
            },
        ),
        # Executions
        Tool(
            name="list_executions",
            description="List agent execution history",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "integer",
                        "description": "Filter by agent ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of executions (default: 20)",
                        "default": 20,
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None,
) -> list[TextContent]:
    """Handle tool execution requests

    Args:
        name: Tool name to execute
        arguments: Tool arguments

    Returns:
        List of text content with results
    """
    args = arguments or {}

    try:
        async with KikiClient(
            KIKI_BASE_URL,
            KIKI_API_KEY,
            KIKI_TIMEOUT,
            AUTH_HEADER_FORMAT,
        ) as client:
            # Agent Management
            if name == "list_agents":
                result = await client.list_agents(
                    agent_type=args.get("agent_type"),
                    status=args.get("status"),
                    page=args.get("page", 1),
                    size=args.get("size", 20),
                )
            elif name == "get_agent":
                result = await client.get_agent(args["agent_id"])
            elif name == "get_agent_stats":
                result = await client.get_agent_stats()
            elif name == "create_agent":
                result = await client.create_agent(
                    name=args["name"],
                    description=args.get("description", ""),
                    agent_type=args["agent_type"],
                    model_name=args["model_name"],
                    system_prompt=args.get("system_prompt", ""),
                    temperature=args.get("temperature", 0.7),
                    max_tokens=args.get("max_tokens", 2000),
                    config=args.get("config"),
                    tool_ids=args.get("tool_ids"),
                )
            elif name == "update_agent":
                result = await client.update_agent(
                    agent_id=args["agent_id"],
                    name=args.get("name"),
                    description=args.get("description"),
                    system_prompt=args.get("system_prompt"),
                    temperature=args.get("temperature"),
                    max_tokens=args.get("max_tokens"),
                    config=args.get("config"),
                    tool_ids=args.get("tool_ids"),
                )
            elif name == "delete_agent":
                await client.delete_agent(args["agent_id"])
                result = {"status": "success", "message": f"Agent {args['agent_id']} deleted"}

            # Chat & Session
            elif name == "chat":
                result = await client.chat(
                    message=args["message"],
                    session_id=args["session_id"],
                    user_id=args.get("user_id"),
                )
            elif name == "get_chat_history":
                result = await client.get_chat_history(args["session_id"])
            elif name == "clear_chat_history":
                result = await client.clear_chat_history(args["session_id"])
            elif name == "get_context_stats":
                result = await client.get_context_stats(args["session_id"])

            # Tools
            elif name == "list_available_tools":
                result = {"tools": await client.list_tools()}

            # Agent Systems
            elif name == "list_agent_systems":
                result = {"systems": await client.list_agent_systems()}
            elif name == "get_agent_system":
                result = await client.get_agent_system(args["system_id"])
            elif name == "delete_agent_system":
                result = await client.delete_agent_system(args["system_id"])

            # Executions
            elif name == "list_executions":
                result = await client.list_executions(
                    agent_id=args.get("agent_id"),
                    limit=args.get("limit", 20),
                )

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

            # Format response
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False, default=str)
            )]

    except Exception as e:
        logger.error(f"Tool execution failed: {name} - {e}")
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def run():
    """Run the MCP server using stdio transport"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="kiki-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """Main entry point"""
    asyncio.run(run())


if __name__ == "__main__":
    main()
