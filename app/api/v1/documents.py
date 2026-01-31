"""文档处理 API 路由

提供文档解析、分块和格式检测的接口。
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.observability.logging import get_logger
from app.schemas.document import (
    FORMAT_DETAILS,
    BatchParseRequest,
    BatchParseResponse,
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentUrlParseRequest,
    SupportedFormatsResponse,
)
from app.schemas.response import ApiResponse
from app.services.document_service import get_document_service

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/parse", response_model=DocumentParseResponse)
async def parse_document(request: DocumentParseRequest) -> DocumentParseResponse:
    """解析文档并分块

    支持多种文档格式：
    - PDF: .pdf
    - Word: .doc, .docx
    - Excel: .xls, .xlsx, .xlsm
    - PowerPoint: .ppt, .pptx
    - Text: .txt
    - Markdown: .md, .markdown
    - Image: .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp (OCR)

    Args:
        request: 文档解析请求

    Returns:
        DocumentParseResponse: 解析结果，包含文档内容和分块

    Raises:
        HTTPException: 请求格式错误

    Examples:
        ```python
        # 从本地文件路径解析
        {
            "file_path": "/path/to/document.pdf",
            "chunk_config": {
                "preset": "default"
            }
        }

        # 从文件内容解析
        {
            "filename": "document.pdf",
            "file_content": "<base64 encoded bytes>",
            "chunk_config": {
                "chunk_size": 1500,
                "chunk_overlap": 300
            }
        }
        ```
    """
    # 验证请求
    if not any([request.file_path, request.file_content]):
        raise HTTPException(
            status_code=400,
            detail="Either file_path or file_content must be provided",
        )

    if request.file_content and not request.filename:
        raise HTTPException(
            status_code=400,
            detail="filename is required when using file_content",
        )

    try:
        service = get_document_service()
        result = await service.parse_document(request)

        logger.info(
            "document_parse_api_success",
            source=result.source,
            format=result.format,
            chunk_count=result.chunk_count,
            success=result.success,
        )

        return result

    except ValueError as e:
        logger.warning(
            "document_parse_api_validation_error",
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "document_parse_api_error",
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Document parse failed: {str(e)}",
        ) from e


@router.post("/parse/upload", response_model=DocumentParseResponse)
async def parse_upload_file(
    file: UploadFile = File(...),  # noqa: B008
    preset: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> DocumentParseResponse:
    """解析上传的文件

    通过 multipart/form-data 上传文件进行解析。

    Args:
        file: 上传的文件
        preset: 预定义配置名称 (default, small, large, code, markdown)
        chunk_size: 自定义块大小
        chunk_overlap: 自定义块重叠大小

    Returns:
        DocumentParseResponse: 解析结果

    Raises:
        HTTPException: 文件处理失败
    """
    from app.schemas.document import ChunkConfig

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    try:
        # 构建分块配置
        chunk_config = None
        if preset or chunk_size is not None or chunk_overlap is not None:
            config_data = {}
            if preset:
                config_data["preset"] = preset
            if chunk_size is not None:
                config_data["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                config_data["chunk_overlap"] = chunk_overlap
            chunk_config = ChunkConfig(**config_data)

        service = get_document_service()
        result = await service.parse_upload_file(file, chunk_config)

        logger.info(
            "document_upload_parse_success",
            filename=file.filename,
            format=result.format,
            chunk_count=result.chunk_count,
            success=result.success,
        )

        return result

    except ValueError as e:
        logger.warning(
            "document_upload_parse_validation_error",
            filename=file.filename,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "document_upload_parse_error",
            filename=file.filename,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"File parse failed: {str(e)}",
        ) from e


@router.post("/parse/url", response_model=DocumentParseResponse)
async def parse_from_url(request: DocumentUrlParseRequest) -> DocumentParseResponse:
    """解析网页内容

    从指定 URL 抓取并解析网页内容。

    Args:
        request: URL 解析请求

    Returns:
        DocumentParseResponse: 解析结果

    Raises:
        HTTPException: URL 处理失败

    Examples:
        ```python
        {
            "url": "https://example.com/article",
            "chunk_config": {
                "preset": "markdown"
            }
        }
        ```
    """
    try:
        service = get_document_service()
        result = await service.parse_from_url(request)

        logger.info(
            "url_parse_api_success",
            url=request.url,
            chunk_count=result.chunk_count,
            success=result.success,
        )

        return result

    except Exception as e:
        logger.error(
            "url_parse_api_error",
            url=request.url,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"URL parse failed: {str(e)}",
        ) from e


@router.get("/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats() -> SupportedFormatsResponse:
    """获取支持的文档格式

    返回所有支持的文档格式及其文件扩展名。

    Returns:
        SupportedFormatsResponse: 支持的格式信息
    """
    service = get_document_service()
    format_info = service.get_supported_formats()

    return SupportedFormatsResponse(
        formats=format_info["formats"],
        extensions=format_info["extensions"],
        format_details=FORMAT_DETAILS,
    )


@router.post("/parse/batch", response_model=BatchParseResponse)
async def parse_batch(request: BatchParseRequest) -> BatchParseResponse:
    """批量解析文档

    一次最多解析 10 个文档。

    Args:
        request: 批量解析请求

    Returns:
        BatchParseResponse: 批量解析结果

    Raises:
        HTTPException: 批量处理失败

    Examples:
        ```python
        {
            "files": [
                {"file_path": "/path/to/doc1.pdf"},
                {"file_path": "/path/to/doc2.docx"}
            ],
            "chunk_config": {
                "preset": "default"
            }
        }
        ```
    """
    if len(request.files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch",
        )

    results = []
    success_count = 0
    failed_count = 0

    service = get_document_service()

    for file_request in request.files:
        # 应用统一分块配置
        if request.chunk_config and not file_request.chunk_config:
            file_request.chunk_config = request.chunk_config

        try:
            result = await service.parse_document(file_request)
            results.append(result)

            if result.success:
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logger.error(
                "batch_parse_item_failed",
                filename=file_request.filename or file_request.file_path,
                error=str(e),
            )
            # 创建失败结果
            results.append(
                DocumentParseResponse(
                    success=False,
                    content="",
                    chunks=[],
                    metadata={},
                    format="unknown",
                    page_count=0,
                    source=file_request.filename or file_request.file_path or "unknown",
                    chunk_count=0,
                    message=str(e),
                )
            )
            failed_count += 1

    logger.info(
        "batch_parse_completed",
        total=len(request.files),
        success_count=success_count,
        failed_count=failed_count,
    )

    return BatchParseResponse(
        total=len(request.files),
        success_count=success_count,
        failed_count=failed_count,
        results=results,
    )


@router.get("/health", response_model=ApiResponse)
async def health_check() -> ApiResponse:
    """文档服务健康检查

    Returns:
        ApiResponse: 健康状态
    """
    return ApiResponse(success=True, message="Document service is healthy")


__all__ = ["router"]
