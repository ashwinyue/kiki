"""内置工具模块

提供常用内置工具，自动注册到全局工具注册表。

包含:
- search_web: Web 搜索工具（DuckDuckGo）
- search_web_tavily: Tavily 搜索工具（支持图像）
- crawl_url: 网页爬取工具（Jina Reader）
- crawl_multiple_urls: 批量网页爬取
- web_fetch: 网页内容摘要（LLM 智能提取）
- python_repl: Python 代码执行工具
- search_database: 数据库搜索工具示例
- get_weather: 天气查询工具示例
- calculate: 数学计算工具示例
"""

from app.agent.tools.builtin.calculation import calculate
from app.agent.tools.builtin.crawl import (
    close_crawler,
    crawl_multiple_urls,
    crawl_url,
)
from app.agent.tools.builtin.database import search_database
from app.agent.tools.builtin.rag import (
    add_knowledge,
    clear_knowledge_base,
    search_knowledge_base,
)
from app.agent.tools.builtin.python_repl import (
    SafePythonREPL,
    get_repl,
    is_python_repl_available,
    python_repl,
)
from app.agent.tools.builtin.search import search_web
from app.agent.tools.builtin.tavily_search import (
    TavilySearchTool,
    search_web_tavily,
    search_web_tavily_sync,
)
from app.agent.tools.builtin.weather import get_weather
from app.agent.tools.builtin.web_fetch import web_fetch

__all__ = [
    # 搜索工具
    "search_web",
    "search_web_tavily",
    "search_web_tavily_sync",
    # 爬虫工具
    "crawl_url",
    "crawl_multiple_urls",
    "close_crawler",
    # 网页内容提取
    "web_fetch",
    # 代码执行
    "python_repl",
    "SafePythonREPL",
    "get_repl",
    "is_python_repl_available",
    # 数据工具
    "search_database",
    "get_weather",
    "calculate",
    # RAG 知识库
    "search_knowledge_base",
    "add_knowledge",
    "clear_knowledge_base",
    # 工具类
    "TavilySearchTool",
]
