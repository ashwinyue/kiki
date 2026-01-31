#!/usr/bin/env python3
"""Test script for Kiki MCP Server

This script tests the MCP server functionality without requiring
an actual MCP client connection.
"""

import asyncio
import json
import os
from typing import Any, Dict

# Import the client
from kiki_mcp_server import KikiClient


async def test_client():
    """Test the KikiClient functionality"""

    # Configuration
    base_url = os.getenv("KIKI_BASE_URL", "http://localhost:8000/api/v1")
    api_key = os.getenv("KIKI_API_KEY", "")

    print(f"Testing Kiki MCP Server")
    print(f"Base URL: {base_url}")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print("-" * 50)

    try:
        async with KikiClient(base_url, api_key) as client:

            # Test 1: List agents
            print("\n[Test 1] Listing agents...")
            try:
                agents = await client.list_agents()
                print(f"  Success! Found {agents.get('total', 0)} agents")
                if agents.get('items'):
                    print(f"  First agent: {agents['items'][0].get('name', 'N/A')}")
            except Exception as e:
                print(f"  Failed: {e}")

            # Test 2: Get agent stats
            print("\n[Test 2] Getting agent stats...")
            try:
                stats = await client.get_agent_stats()
                print(f"  Success! Total agents: {stats.get('total_agents', 0)}")
                print(f"  Active agents: {stats.get('active_agents', 0)}")
            except Exception as e:
                print(f"  Failed: {e}")

            # Test 3: List available tools
            print("\n[Test 3] Listing available tools...")
            try:
                tools = await client.list_tools()
                print(f"  Success! Found {len(tools)} tools")
                for tool in tools[:5]:  # Show first 5
                    print(f"    - {tool.get('name', 'N/A')}")
            except Exception as e:
                print(f"  Failed: {e}")

            # Test 4: List agent systems
            print("\n[Test 4] Listing agent systems...")
            try:
                systems = await client.list_agent_systems()
                print(f"  Success! Found {len(systems)} agent systems")
            except Exception as e:
                print(f"  Failed: {e}")

            # Test 5: Chat (with a test session)
            print("\n[Test 5] Testing chat endpoint...")
            try:
                response = await client.chat(
                    message="Hello!",
                    session_id="test-mcp-session"
                )
                print(f"  Success! Response: {response.get('content', 'N/A')[:100]}...")
            except Exception as e:
                print(f"  Failed: {e}")

        print("\n" + "=" * 50)
        print("All tests completed!")

    except RuntimeError as e:
        print(f"\nConnection Error: {e}")
        print("\nMake sure the Kiki server is running:")
        print("  cd /path/to/kiki")
        print("  uv run uvicorn app.main:app --reload")


async def test_tools_definition():
    """Test that tools are properly defined"""
    from kiki_mcp_server import app

    print("\n[Tools Definition] Available MCP tools:")
    print("-" * 50)

    # Get tools
    tools = await app._handlers["tools/list"]()

    for tool in tools:
        print(f"\n{tool.name}:")
        print(f"  Description: {tool.description}")
        if tool.inputSchema.get("required"):
            print(f"  Required: {', '.join(tool.inputSchema['required'])}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("Kiki MCP Server Test Suite")
    print("=" * 50)

    # Run client tests
    asyncio.run(test_client())

    # Run tools definition test
    asyncio.run(test_tools_definition())


if __name__ == "__main__":
    main()
