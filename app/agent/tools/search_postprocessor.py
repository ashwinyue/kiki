"""搜索结果后处理器

处理搜索结果：去重、过滤、清理、排序。
参考 DeerFlow 的 SearchResultPostProcessor 设计。

使用示例:
```python
from app.agent.tools.search_postprocessor import SearchResultPostProcessor

processor = SearchResultPostProcessor(
    min_score_threshold=0.5,
    max_content_length=2000,
)

cleaned_results = processor.process_results(raw_results)
```
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from app.observability.log_sanitizer import sanitize_log_input, sanitize_tool_name

logger = logging.getLogger(__name__)


class SearchResultPostProcessor:
    """搜索结果后处理器

    处理搜索结果：
    1. 去重
    2. 过滤低质量结果
    3. 清理 base64 图片
    4. 截断过长内容
    5. 按分数排序
    """

    # base64 图片模式
    base64_pattern = r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+"

    def __init__(
        self,
        min_score_threshold: float = 0.0,
        max_content_length: int = 5000,
    ):
        """初始化后处理器

        Args:
            min_score_threshold: 最低相关性分数阈值
            max_content_length: 每页最大内容长度
        """
        self.min_score_threshold = min_score_threshold
        self.max_content_length = max_content_length

    def process_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """处理搜索结果

        Args:
            results: 原始搜索结果列表

        Returns:
            处理后的结果列表
        """
        if not results:
            return []

        logger.debug(
            "search_postprocessing_start",
            result_count=len(results),
            min_score=self.min_score_threshold,
            max_length=self.max_content_length,
        )

        # 单次循环处理，提高效率
        cleaned_results = []
        seen_urls = set()

        for result in results:
            # 1. 去重
            cleaned_result = self._remove_duplicate(result, seen_urls)
            if not cleaned_result:
                continue

            # 2. 过滤低质量结果
            if not self._check_quality(cleaned_result):
                continue

            # 3. 清理 base64 图片
            cleaned_result = self._remove_base64_images(cleaned_result)
            if not cleaned_result:
                continue

            # 4. 截断过长内容
            if self.max_content_length > 0:
                cleaned_result = self._truncate_content(cleaned_result)

            if cleaned_result:
                cleaned_results.append(cleaned_result)

        # 5. 按分数排序
        sorted_results = sorted(
            cleaned_results,
            key=lambda x: float(x.get("score", 0)),
            reverse=True,
        )

        logger.info(
            "search_postprocessing_complete",
            original_count=len(results),
            final_count=len(sorted_results),
        )

        return sorted_results

    def _remove_duplicate(
        self, result: dict[str, Any], seen_urls: set
    ) -> dict[str, Any] | None:
        """移除重复结果

        Args:
            result: 搜索结果
            seen_urls: 已见 URL 集合

        Returns:
            去重后的结果，或 None（如果是重复）
        """
        url = result.get("url") or result.get("href")
        if url and url in seen_urls:
            logger.debug("duplicate_removed", url=sanitize_log_input(url, max_length=100))
            return None

        if url:
            seen_urls.add(url)

        return result.copy()

    def _check_quality(self, result: dict[str, Any]) -> bool:
        """检查结果质量

        Args:
            result: 搜索结果

        Returns:
            是否通过质量检查
        """
        # 分数过滤
        if (
            self.min_score_threshold > 0
            and result.get("type") == "page"
            and float(result.get("score", 0)) < self.min_score_threshold
        ):
            logger.debug(
                "low_score_filtered",
                url=sanitize_log_input(result.get("url", ""), max_length=100),
                score=result.get("score"),
            )
            return False

        return True

    def _remove_base64_images(
        self, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """移除 base64 编码的图片

        Args:
            result: 搜索结果

        Returns:
            清理后的结果，或 None（如果结果无效）
        """
        cleaned_result = result.copy()
        result_type = cleaned_result.get("type", "page")

        if result_type == "page":
            cleaned_result = self._clean_page_images(cleaned_result)
        elif result_type == "image":
            cleaned_result = self._clean_image_data(cleaned_result)
            if not cleaned_result:
                return None

        return cleaned_result

    def _clean_page_images(self, result: dict[str, Any]) -> dict[str, Any]:
        """清理页面类型结果中的图片

        Args:
            result: 搜索结果

        Returns:
            清理后的结果
        """
        if "content" in result:
            original_content = result["content"]
            cleaned_content = re.sub(self.base64_pattern, " ", original_content)
            result["content"] = cleaned_content

            # 记录显著的内容减少
            if len(cleaned_content) < len(original_content) * 0.8:
                logger.debug(
                    "base64_removed_from_content",
                    url=sanitize_log_input(result.get("url", ""), max_length=100),
                    removed_chars=len(original_content) - len(cleaned_content),
                )

        if "raw_content" in result:
            original_raw = result["raw_content"]
            cleaned_raw = re.sub(self.base64_pattern, " ", original_raw)
            result["raw_content"] = cleaned_raw

        return result

    def _clean_image_data(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """清理图片类型结果中的 base64 数据

        Args:
            result: 搜索结果

        Returns:
            清理后的结果，或 None（如果无效）
        """
        image_url = result.get("image_url", "")

        if isinstance(image_url, str) and "data:image" in image_url:
            cleaned_url = re.sub(self.base64_pattern, " ", image_url)

            # 检查清理后是否有效
            if not cleaned_url.strip() or not cleaned_url.startswith("http"):
                logger.debug(
                    "invalid_image_after_cleaning",
                    original_url=sanitize_log_input(image_url[:50], max_length=50),
                )
                return None

            result["image_url"] = cleaned_url
            logger.debug(
                "base64_removed_from_image_url",
                original_url=sanitize_log_input(image_url[:50], max_length=50),
            )

        return result

    def _truncate_content(self, result: dict[str, Any]) -> dict[str, Any]:
        """截断过长内容

        Args:
            result: 搜索结果

        Returns:
            截断后的结果
        """
        # 截断 content
        if "content" in result:
            content = result["content"]
            if len(content) > self.max_content_length:
                result["content"] = content[: self.max_content_length] + "..."
                logger.debug(
                    "content_truncated",
                    url=sanitize_log_input(result.get("url", ""), max_length=100),
                    original_length=len(content),
                )

        # 截断 raw_content（可以稍长）
        if "raw_content" in result:
            raw_content = result["raw_content"]
            max_raw = self.max_content_length * 2
            if len(raw_content) > max_raw:
                result["raw_content"] = raw_content[:max_raw] + "..."
                logger.debug(
                    "raw_content_truncated",
                    url=sanitize_log_input(result.get("url", ""), max_length=100),
                    original_length=len(raw_content),
                )

        # 截断 image_description
        if "image_description" in result:
            desc = result["image_description"]
            if len(desc) > self.max_content_length:
                result["image_description"] = desc[: self.max_content_length] + "..."

        return result


def is_pdf_url(url: str | None) -> bool:
    """检查 URL 是否指向 PDF 文件

    Args:
        url: 要检查的 URL

    Returns:
        是否为 PDF URL
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        return parsed.path.lower().endswith(".pdf")
    except Exception:
        return False


def is_valid_url(url: str | None) -> bool:
    """检查 URL 是否有效

    Args:
        url: 要检查的 URL

    Returns:
        URL 是否有效
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and parsed.netloc
    except Exception:
        return False


def extract_domain(url: str | None) -> str | None:
    """从 URL 中提取域名

    Args:
        url: URL 字符串

    Returns:
        域名，或 None
    """
    if not url:
        return None

    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def normalize_url(url: str) -> str:
    """规范化 URL（移除跟踪参数等）

    Args:
        url: 原始 URL

    Returns:
        规范化后的 URL
    """
    try:
        parsed = urlparse(url)
        # 移除常见的跟踪参数
        # 简化实现：保持基本 URL
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            # 保留部分重要查询参数
            important_params = ["id", "v", "p"]
            params = []
            for param in parsed.query.split("&"):
                key = param.split("=")[0] if "=" in param else param
                if key in important_params:
                    params.append(param)
            if params:
                normalized += "?" + "&".join(params)

        return normalized
    except Exception:
        return url
