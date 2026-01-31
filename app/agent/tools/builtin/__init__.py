"""内置工具模块

提供常用内置工具，自动注册到全局工具注册表。

包含:
- search_web: Web 搜索工具（DuckDuckGo）
- search_web_tavily: Tavily 搜索工具（支持图像）
- search_wikipedia: Wikipedia 百科搜索
- search_arxiv: ArXiv 学术搜索
- search_academic: 多源学术搜索
- crawl_url: 网页爬取工具（Jina Reader）
- crawl_multiple_urls: 批量网页爬取
- web_fetch: 网页内容摘要（LLM 智能提取）
- grep_chunks: 知识库全文搜索
- query_knowledge_graph: 知识图谱查询
- search_entity: 知识图谱实体搜索
- list_knowledge_chunks: 列出知识库 chunks
- get_document_info: 获取文档信息
- data_analysis: 数据分析（CSV/Excel SQL 查询）
- database_query: 数据库查询（只读）
- data_schema: 数据模式分析
- python_repl: Python 代码执行工具（数据分析）
- search_database: 数据库搜索工具示例
- get_weather: 天气查询工具示例
- calculate: 数学计算工具示例
"""

from app.agent.tools.builtin.academic import (
    get_available_sources,
    is_arxiv_available,
    is_wikipedia_available,
    search_academic,
    search_arxiv,
    search_wikipedia,
)
from app.agent.tools.builtin.calculation import calculate
from app.agent.tools.builtin.crawl import (
    close_crawler,
    crawl_multiple_urls,
    crawl_url,
)
from app.agent.tools.builtin.data_analysis import data_analysis
from app.agent.tools.builtin.data_schema import data_schema, data_summary
from app.agent.tools.builtin.database import search_database
from app.agent.tools.builtin.database_query import (
    database_describe,
    database_query,
    database_tables,
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

# 新增工具
from app.agent.tools.builtin.get_document_info import get_document_info
from app.agent.tools.builtin.grep_chunks import grep_chunks
from app.agent.tools.builtin.list_knowledge_chunks import list_knowledge_chunks
from app.agent.tools.builtin.query_knowledge_graph import query_knowledge_graph
from app.agent.tools.builtin.search_entity import search_entity
from app.agent.tools.builtin.web_fetch import web_fetch

__all__ = [
    # 搜索工具
    "search_web",
    "search_web_tavily",
    "search_web_tavily_sync",
    "search_wikipedia",
    "search_arxiv",
    "search_academic",
    # 爬虫工具
    "crawl_url",
    "crawl_multiple_urls",
    "close_crawler",
    # 网页内容提取
    "web_fetch",
    # 知识库搜索
    "grep_chunks",
    "query_knowledge_graph",
    "search_entity",
    "list_knowledge_chunks",
    "get_document_info",
    # 数据分析工具
    "data_analysis",
    "data_schema",
    "data_summary",
    # 数据库工具
    "database_query",
    "database_tables",
    "database_describe",
    # 代码执行
    "python_repl",
    "SafePythonREPL",
    "get_repl",
    "is_python_repl_available",
    # 数据工具
    "search_database",
    "get_weather",
    "calculate",
    # 工具类
    "TavilySearchTool",
    # 可用性检查
    "is_wikipedia_available",
    "is_arxiv_available",
    "get_available_sources",
]
