"""网页内容摘要工具

获取网页内容并使用 LLM 提取关键信息。
对齐 WeKnora99 ToolWebFetch。
"""

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel

from app.agent.tools.search_postprocessor import is_valid_url
from app.observability.log_sanitizer import sanitize_log_input
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认配置
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_LENGTH = 50000


class WebFetchError(Exception):
    """网页获取错误"""
    pass


class WebFetchConfig(BaseModel):
    """网页获取配置"""

    timeout: float = DEFAULT_TIMEOUT
    max_content_length: int = DEFAULT_MAX_LENGTH
    user_agent: str = (
        "Mozilla/5.0 (compatible; KikiAgent/1.0; +https://github.com/kiki)"
    )


async def _fetch_url(
    url: str,
    config: WebFetchConfig | None = None,
) -> str:
    """获取网页内容

    Args:
        url: 网页 URL
        config: 配置

    Returns:
        网页内容
    """
    config = config or WebFetchConfig()

    if not is_valid_url(url):
        raise WebFetchError(f"Invalid URL: {url}")

    safe_url = sanitize_log_input(url, max_length=100)
    logger.info("web_fetch_url", url=safe_url)

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            headers = {"User-Agent": config.user_agent}
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            content = response.text

            # 截断过长内容
            if len(content) > config.max_content_length:
                content = content[: config.max_content_length] + "\n\n... (content truncated)"

            logger.info(
                "web_fetch_success",
                url=safe_url,
                content_length=len(content),
            )

            return content

    except httpx.HTTPStatusError as e:
        logger.error(
            "web_fetch_http_error",
            url=safe_url,
            status_code=e.response.status_code,
        )
        raise WebFetchError(f"HTTP {e.response.status_code}: {e.response.reason}")

    except httpx.TimeoutException:
        logger.error("web_fetch_timeout", url=safe_url)
        raise WebFetchError(f"Timeout after {config.timeout}s")

    except Exception as e:
        logger.error("web_fetch_failed", url=safe_url, error=str(e))
        raise WebFetchError(f"Failed to fetch {url}: {str(e)}")


async def _summarize_content(
    content: str,
    prompt: str,
    url: str,
) -> str:
    """使用 LLM 摘要内容

    Args:
        content: 网页内容
        prompt: 摘要提示词
        url: 原始 URL

    Returns:
        摘要结果
    """
    try:
        from app.llm.service import get_llm_service

        llm_service = get_llm_service()

        messages = [
            {
                "role": "system",
                "content": f"""你是一个专业的网页内容分析助手。
用户请求从以下网页中提取关键信息：

URL: {url}

请根据用户的提示词，从网页内容中提取相关信息。
如果网页内容与用户请求无关，请说明这一点。
请直接输出提取的信息，不要添加额外的解释。""",
            },
            {
                "role": "user",
                "content": f"""网页内容：
---
{content[:10000]}  # 限制内容长度
---

请提取以下信息：
{prompt}

如果内容过长，我已截取前 10000 字符。如需查看完整内容，请直接访问 URL。""",
            },
        ]

        response = await llm_service.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        return response.get("content", "无法生成摘要")

    except ImportError:
        # 如果 LLM 服务不可用，返回原始内容的前 2000 字符
        logger.warning("llm_service_unavailable")
        return content[:2000] + ("\n\n... (内容已截断)" if len(content) > 2000 else "")

    except Exception as e:
        logger.error("summarize_failed", error=str(e))
        # 出错时返回原始内容
        return content[:2000] + ("\n\n... (内容已截断)" if len(content) > 2000 else "")


def _format_result(
    url: str,
    summary: str,
    raw_content: str | None = None,
) -> str:
    """格式化结果

    Args:
        url: 原始 URL
        summary: 摘要内容
        raw_content: 原始内容（可选）

    Returns:
        格式化的结果
    """
    parts = [
        f"## 网页内容摘要",
        f"**URL**: {url}",
        "",
        "### 提取的信息",
        summary,
    ]

    if raw_content and len(raw_content) > 10000:
        parts.extend([
            "",
            "---",
            f"*注意：原始内容较长，已截取前 10000 字符。完整内容请访问 [原始链接]({url})。*",
        ])

    return "\n".join(parts)


@tool
async def web_fetch(
    url: str,
    prompt: str = "Extract the key information from this webpage, focusing on main points and important details.",
) -> str:
    """获取网页内容并提取关键信息

    使用 LLM 对网页内容进行智能摘要，提取用户关心的关键信息。

    Args:
        url: 要获取的网页 URL
        prompt: 提取信息的提示词（英文效果更好）

    Returns:
        格式化的网页内容摘要

    Examples:
        ```python
        # 提取技术文档的关键信息
        result = await web_fetch(
            "https://docs.python.org/3/",
            "What are the main features of Python 3?"
        )

        # 提取新闻要点
        result = await web_fetch(
            "https://news.example.com/article",
            "What is this news article about? Who, what, when, where, why?"
        )
        ```
    """
    # 参数验证
    if not url:
        return "错误: 请提供有效的 URL"

    if not is_valid_url(url):
        safe_url = sanitize_log_input(url, max_length=50)
        logger.warning("invalid_url", url=safe_url)
        return f"错误: 无效的 URL 格式: {url}"

    logger.info(
        "web_fetch_start",
        url=sanitize_log_input(url, max_length=100),
        prompt_length=len(prompt),
    )

    try:
        # 获取网页内容
        config = WebFetchConfig()
        raw_content = await _fetch_url(url, config)

        if not raw_content or raw_content.strip() == "":
            return f"错误: 无法获取网页内容（URL: {url}）"

        # 使用 LLM 摘要
        summary = await _summarize_content(raw_content, prompt, url)

        # 格式化结果
        result = _format_result(url, summary, raw_content)

        logger.info(
            "web_fetch_complete",
            url=sanitize_log_input(url, max_length=100),
            summary_length=len(summary),
        )

        return result

    except WebFetchError as e:
        logger.error("web_fetch_failed", url=sanitize_log_input(url, max_length=100), error=str(e))
        return f"错误: {str(e)}"

    except Exception as e:
        logger.exception("web_fetch_error", url=sanitize_log_input(url, max_length=100), error=str(e))
        return f"错误: 获取网页失败: {str(e)}"


__all__ = ["web_fetch"]
