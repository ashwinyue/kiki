"""网页爬取工具

使用 Jina Reader API 爬取网页内容，返回 Markdown 格式。
参考 DeerFlow 的爬虫设计。

Jina Reader 优势：
- 免费使用
- 自动提取主要内容
- 返回 Markdown 格式
- 无需维护爬虫逻辑

使用示例:
```python
from app.agent.tools.builtin.crawl import crawl_url

content = await crawl_url("https://example.com")
```
"""

import asyncio
import logging
from typing import Literal

import httpx
from langchain_core.tools import tool

from app.agent.tools.search_postprocessor import is_pdf_url, is_valid_url
from app.observability.log_sanitizer import sanitize_log_input
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Jina Reader API 端点
JINA_READER_API = "https://r.jina.ai/http://"

# 默认超时时间（秒）
DEFAULT_TIMEOUT = 30.0
MAX_CONTENT_LENGTH = 50000


class CrawlerError(Exception):
    """爬虫错误"""
    pass


class CrawlerConfig:
    """爬虫配置"""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_content_length: int = MAX_CONTENT_LENGTH,
        user_agent: str | None = None,
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; KikiAgent/1.0; +https://github.com/kiki)"
        )


class JinaReader:
    """Jina Reader 客户端

    使用 Jina Reader API 爬取网页内容。
    """

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端

        Returns:
            httpx.AsyncClient 实例
        """
        if self._client is None:
            timeout = httpx.Timeout(self.config.timeout)
            headers = {
                "User-Agent": self.config.user_agent,
            }
            self._client = httpx.AsyncClient(timeout=timeout, headers=headers)
        return self._client

    async def crawl(
        self,
        url: str,
        return_format: Literal["markdown", "html", "text"] = "markdown",
    ) -> str:
        """爬取网页内容

        Args:
            url: 要爬取的 URL
            return_format: 返回格式（markdown/html/text）

        Returns:
            爬取的内容

        Raises:
            CrawlerError: 爬取失败
        """
        if not is_valid_url(url):
            raise CrawlerError(f"Invalid URL: {url}")

        if is_pdf_url(url):
            return self._format_pdf_response(url)

        safe_url = sanitize_log_input(url, max_length=100)
        logger.info("crawling_start", url=safe_url, format=return_format)

        try:
            client = await self._get_client()

            # 构建 Jina Reader API URL
            api_url = f"{JINA_READER_API}{url}"

            response = await client.get(api_url)
            response.raise_for_status()

            content = response.text

            # 截断过长内容
            if len(content) > self.config.max_content_length:
                content = content[: self.config.max_content_length] + "\n\n... (content truncated)"

            logger.info(
                "crawling_complete",
                url=safe_url,
                content_length=len(content),
            )

            return content

        except httpx.HTTPStatusError as e:
            logger.error(
                "crawling_http_error",
                url=safe_url,
                status_code=e.response.status_code,
            )
            raise CrawlerError(f"HTTP {e.response.status_code}: {e.response.reason}")
        except httpx.TimeoutException:
            logger.error("crawling_timeout", url=safe_url)
            raise CrawlerError(f"Timeout after {self.config.timeout}s")
        except Exception as e:
            logger.error("crawling_failed", url=safe_url, error=str(e))
            raise CrawlerError(f"Crawling failed: {str(e)}")

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @staticmethod
    def _format_pdf_response(url: str) -> str:
        """格式化 PDF 响应

        Args:
            url: PDF URL

        Returns:
            格式化的响应
        """
        return f"# PDF Document\n\n**URL**: {url}\n\nPDF files cannot be directly crawled. Please download and view the PDF manually."


# 全局爬虫实例
_crawler: JinaReader | None = None


def get_crawler() -> JinaReader:
    """获取全局爬虫实例

    Returns:
        JinaReader 实例
    """
    global _crawler
    if _crawler is None:
        _crawler = JinaReader()
    return _crawler


@tool
async def crawl_url(
    url: str,
    max_length: int = 10000,
) -> str:
    """爬取网页内容，返回 Markdown 格式

    使用 Jina Reader API（免费）爬取网页内容。
    自动提取主要内容，移除广告和导航元素。

    Args:
        url: 要爬取的网页 URL
        max_length: 最大返回内容长度（字符数）

    Returns:
        Markdown 格式的网页内容

    Examples:
        >>> content = await crawl_url("https://example.com")

        >>> # 支持的 URL 格式
        >>> await crawl_url("https://example.com/article")
        >>> await crawl_url("http://example.com")
    """
    safe_url = sanitize_log_input(url, max_length=100)

    # 验证 URL
    if not is_valid_url(url):
        logger.warning("invalid_url", url=safe_url)
        return f"Error: Invalid URL: {url}"

    # 检查 PDF
    if is_pdf_url(url):
        logger.info("pdf_url_detected", url=safe_url)
        return f"# PDF Document\n\n**URL**: {url}\n\nPDF files cannot be directly crawled. Please download and view the PDF manually."

    logger.info("crawl_url_start", url=safe_url, max_length=max_length)

    try:
        # 获取爬虫实例
        crawler = JinaReader(
            CrawlerConfig(max_content_length=max_length)
        )

        # 爬取内容
        content = await crawler.crawl(url, return_format="markdown")

        logger.info(
            "crawl_url_success",
            url=safe_url,
            content_length=len(content),
        )

        return content

    except CrawlerError as e:
        logger.error("crawl_url_failed", url=safe_url, error=str(e))
        return f"Error: Failed to crawl {url}: {str(e)}"

    except Exception as e:
        logger.error("crawl_url_error", url=safe_url, error=str(e))
        return f"Error: Unexpected error crawling {url}: {str(e)}"


@tool
async def crawl_multiple_urls(
    urls: list[str],
    max_length: int = 5000,
) -> str:
    """批量爬取多个网页内容

    Args:
        urls: 要爬取的 URL 列表
        max_length: 每个网页的最大内容长度

    Returns:
        所有网页内容的合并结果

    Examples:
        >>> urls = ["https://example.com/1", "https://example.com/2"]
        >>> content = await crawl_multiple_urls(urls)
    """
    if not urls:
        return "Error: No URLs provided"

    if len(urls) > 10:
        return "Error: Maximum 10 URLs allowed per request"

    logger.info(
        "crawl_multiple_start",
        count=len(urls),
        max_length=max_length,
    )

    results = []
    for i, url in enumerate(urls, 1):
        safe_url = sanitize_log_input(url, max_length=100)
        logger.info("crawl_multiple_progress", url=safe_url, current=i, total=len(urls))

        try:
            content = await crawl_url(url, max_length=max_length)
            results.append(f"## {i}. {url}\n\n{content}\n\n")
        except Exception as e:
            logger.error("crawl_multiple_failed", url=safe_url, error=str(e))
            results.append(f"## {i}. {url}\n\nError: {str(e)}\n\n")

    return "\n".join(results)


async def close_crawler():
    """关闭全局爬虫实例

    在应用关闭时调用。
    """
    global _crawler
    if _crawler:
        await _crawler.close()
        _crawler = None
