"""统一文档服务

提供文档解析、分块和格式检测的统一接口。
"""

import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.observability.logging import get_logger
from app.schemas.document import (
    ChunkConfig as SchemaChunkConfig,
)
from app.schemas.document import (
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentUrlParseRequest,
)
from app.services.document_loaders import (
    DocumentFormat,
    LoaderFactory,
    load_document,
)
from app.services.document_splitter import (
    ChunkConfig,
    DocumentSplitter,
    get_preset_config,
)

logger = get_logger(__name__)


class DocumentService:
    """文档处理服务

    提供统一的文档解析、分块和格式处理接口。
    """

    def __init__(self) -> None:
        """初始化文档服务"""
        logger.info("document_service_initialized")

    async def parse_document(
        self,
        request: DocumentParseRequest,
    ) -> DocumentParseResponse:
        """解析文档并分块

        Args:
            request: 解析请求

        Returns:
            DocumentParseResponse: 解析结果

        Raises:
            ValueError: 文件格式不支持
            RuntimeError: 文档处理失败
        """
        try:
            # 1. 确定文件来源
            if request.file_path:
                source = request.file_path
            elif request.file_content and request.filename:
                # 从内容创建临时文件
                source = await self._save_temp_file(
                    request.file_content,
                    request.filename,
                )
            else:
                raise ValueError(
                    "Either file_path or file_content+filename must be provided"
                )

            # 2. 获取分块配置
            chunk_config = self._get_chunk_config(request.chunk_config)

            # 3. 加载文档
            format = (
                DocumentFormat(request.format)
                if request.format
                else None
            )
            load_result = await load_document(source, format)

            # 4. 分割文档
            splitter = DocumentSplitter(chunk_config)
            chunks = splitter.split_text(
                load_result.content,
                metadata={
                    "source": load_result.source,
                    "format": load_result.format.value,
                    **load_result.metadata,
                },
            )

            # 5. 构建响应
            response = DocumentParseResponse(
                success=True,
                content=load_result.content,
                chunks=[
                    {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                    }
                    for chunk in chunks
                ],
                metadata=load_result.metadata,
                format=load_result.format.value,
                page_count=load_result.page_count,
                source=load_result.source,
                chunk_count=len(chunks),
                message="Document parsed successfully",
            )

            logger.info(
                "document_parsed",
                source=source,
                format=load_result.format.value,
                chunk_count=len(chunks),
                content_length=len(load_result.content),
            )

            return response

        except ValueError as e:
            logger.error(
                "document_parse_validation_error",
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "document_parse_failed",
                error=str(e),
            )
            return DocumentParseResponse(
                success=False,
                content="",
                chunks=[],
                metadata={},
                format="unknown",
                page_count=0,
                source=request.file_path or request.filename or "",
                chunk_count=0,
                message=f"Parse failed: {str(e)}",
            )

    async def parse_from_url(
        self,
        request: DocumentUrlParseRequest,
    ) -> DocumentParseResponse:
        """解析网页内容

        Args:
            request: URL 解析请求

        Returns:
            DocumentParseResponse: 解析结果
        """
        try:
            # 1. 获取分块配置
            chunk_config = self._get_chunk_config(request.chunk_config)

            # 2. 加载网页
            load_result = await load_document(request.url, DocumentFormat.WEB)

            # 3. 分割文档
            splitter = DocumentSplitter(chunk_config)
            chunks = splitter.split_text(
                load_result.content,
                metadata={
                    "source": load_result.source,
                    "format": "web",
                    **load_result.metadata,
                },
            )

            # 4. 构建响应
            response = DocumentParseResponse(
                success=True,
                content=load_result.content,
                chunks=[
                    {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                    }
                    for chunk in chunks
                ],
                metadata=load_result.metadata,
                format=load_result.format.value,
                page_count=load_result.page_count,
                source=load_result.source,
                chunk_count=len(chunks),
                message="URL content parsed successfully",
            )

            logger.info(
                "url_parsed",
                url=request.url,
                chunk_count=len(chunks),
                content_length=len(load_result.content),
            )

            return response

        except Exception as e:
            logger.error(
                "url_parse_failed",
                url=request.url,
                error=str(e),
            )
            return DocumentParseResponse(
                success=False,
                content="",
                chunks=[],
                metadata={},
                format="web",
                page_count=0,
                source=request.url,
                chunk_count=0,
                message=f"URL parse failed: {str(e)}",
            )

    async def parse_upload_file(
        self,
        file: UploadFile,
        chunk_config: SchemaChunkConfig | None = None,
    ) -> DocumentParseResponse:
        """解析上传的文件

        Args:
            file: 上传的文件
            chunk_config: 分块配置

        Returns:
            DocumentParseResponse: 解析结果
        """
        # 读取文件内容
        content = await file.read()

        # 构建请求
        request = DocumentParseRequest(
            filename=file.filename or "unknown",
            file_content=content,
            chunk_config=chunk_config,
        )

        return await self.parse_document(request)

    def get_supported_formats(self) -> dict[str, list[str]]:
        """获取支持的格式

        Returns:
            格式信息字典
        """
        return {
            "formats": LoaderFactory.get_supported_formats(),
            "extensions": DocumentFormat.get_supported_extensions(),
        }

    def _get_chunk_config(
        self,
        config: SchemaChunkConfig | None,
    ) -> ChunkConfig:
        """获取分块配置

        Args:
            config: Schema 分块配置

        Returns:
            ChunkConfig 实例
        """
        if config is None:
            return ChunkConfig()

        if config.preset:
            return get_preset_config(config.preset)

        return ChunkConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    async def _save_temp_file(
        self,
        content: bytes,
        filename: str,
    ) -> str:
        """保存内容到临时文件

        Args:
            content: 文件内容
            filename: 文件名

        Returns:
            临时文件路径
        """
        # 创建临时文件
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp_file:
            tmp_file.write(content)
            return tmp_file.name

    async def close(self) -> None:
        """关闭服务，清理资源"""
        logger.info("document_service_closed")


# 全局服务实例
_document_service: DocumentService | None = None


def get_document_service() -> DocumentService:
    """获取文档服务实例

    Returns:
        DocumentService 实例
    """
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


async def close_document_service() -> None:
    """关闭文档服务"""
    global _document_service
    if _document_service is not None:
        await _document_service.close()
        _document_service = None


__all__ = [
    "DocumentService",
    "get_document_service",
    "close_document_service",
]
