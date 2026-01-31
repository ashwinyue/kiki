#!/usr/bin/env python3
"""Kiki MCP Server Main Entry Point

This file provides a unified entry point for starting the Kiki MCP server.

Usage:
    python main.py                    # Start with default config
    python main.py --check-only       # Check environment only
    python main.py --verbose          # Enable verbose logging
    python main.py --version          # Show version info

Environment Variables:
    KIKI_BASE_URL    Kiki API base URL (default: http://localhost:8000/api/v1)
    KIKI_API_KEY     Kiki API key for authentication (optional)
    KIKI_TIMEOUT     Request timeout in seconds (default: 120)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def setup_environment():
    """Set up Python path and environment"""
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import mcp
        import httpx
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def check_environment_variables():
    """Check and display environment configuration"""
    base_url = os.getenv("KIKI_BASE_URL", "http://localhost:8000/api/v1")
    api_key = os.getenv("KIKI_API_KEY", "")
    timeout = os.getenv("KIKI_TIMEOUT", "120")

    print("=== Kiki MCP Server Environment Check ===")
    print(f"Base URL: {base_url}")
    print(f"API Key: {'Set' if api_key else 'Not set (optional)'}")
    print(f"Timeout: {timeout}s")
    print("=" * 45)

    if not api_key:
        print("Note: API_KEY is not set. This is optional unless your Kiki server requires authentication.")

    return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kiki MCP Server - Model Context Protocol server for Kiki Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with default configuration
  python main.py --check-only       # Check environment only
  python main.py --verbose          # Enable verbose logging

Environment Variables:
  KIKI_BASE_URL    Kiki API base URL (default: http://localhost:8000/api/v1)
  KIKI_API_KEY     Kiki API key for authentication (optional)
  KIKI_TIMEOUT     Request timeout in seconds (default: 120)
        """,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment configuration, don't start server"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Kiki MCP Server 1.0.0"
    )

    return parser.parse_args()


async def main():
    """Main async function"""
    args = parse_arguments()

    # Set up environment
    setup_environment()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check environment variables
    check_environment_variables()

    # Exit if only checking environment
    if args.check_only:
        print("Environment check complete.")
        return

    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("Verbose logging enabled.")

    try:
        print("Starting Kiki MCP Server...")

        # Import and run the server
        from kiki_mcp_server import run

        await run()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all files are in the correct location.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Server error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def sync_main():
    """Synchronous main for entry_points"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
