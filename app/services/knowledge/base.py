"""知识库服务

提供知识库和知识条目的业务逻辑
"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.builtin.crawl import CrawlerConfig, JinaReader
from app.models.knowledge import KnowledgeBase
from app.observability.logging import get_logger
from app.repositories.base import PaginationParams
from app.repositories.knowledge import (
    ChunkRepository,
    KnowledgeBaseRepository,
    KnowledgeRepository,
)
from app.schemas.knowledge import (
    HybridSearchRequest,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeUpdate,
)
from app.services.search.hybrid_search import (
    HybridSearchService,
    SearchParams,
)

logger = get_logger(__name__)

# 支持的文件类型和对应的加载器
SUPPORTED_FILE_TYPES = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
}


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower()


def is_supported_file_type(filename: str) -> bool:
    """检查文件类型是否支持"""
    ext = get_file_extension(filename)
    return ext in SUPPORTED_FILE_TYPES


def get_file_loader(file_path: str) -> type | None:
    """根据文件扩展名获取文档加载器"""
    ext = get_file_extension(file_path)
    return SUPPORTED_FILE_TYPES.get(ext)


class KnowledgeBaseService:
    """知识库服务"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._kb_repo: KnowledgeBaseRepository | None = None

    @property
    def kb_repo(self) -> KnowledgeBaseRepository:
        """延迟初始化知识库仓储"""
        if self._kb_repo is None:
            self._kb_repo = KnowledgeBaseRepository(self.session)
        return self._kb_repo

    async def create_knowledge_base(
        self, data: KnowledgeBaseCreate, tenant_id: int
    ) -> KnowledgeBase:
        """创建知识库

        Args:
            data: 创建请求
            tenant_id: 租户 ID

        Returns:
            创建的知识库
        """
        # 转换数据
        create_data: dict = data.model_dump(exclude_unset=True)
        if data.chunking_config:
            create_data["chunking_config"] = data.chunking_config.model_dump()

        kb = await self.kb_repo.create_with_tenant(create_data, tenant_id)

        logger.info(
            "knowledge_base_created",
            kb_id=kb.id,
            tenant_id=tenant_id,
            name=kb.name,
        )
        return kb

    async def get_knowledge_base(
        self, kb_id: str, tenant_id: int
    ) -> KnowledgeBase | None:
        """获取知识库详情

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            知识库实例
        """
        return await self.kb_repo.get_by_tenant(kb_id, tenant_id)

    async def list_knowledge_bases(
        self, tenant_id: int, params: PaginationParams
    ) -> list[KnowledgeBase]:
        """查询知识库列表

        Args:
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            知识库列表
        """
        result = await self.kb_repo.list_paginated_by_tenant(tenant_id, params)
        return result.items

    async def update_knowledge_base(
        self, kb_id: str, data: KnowledgeBaseUpdate, tenant_id: int
    ) -> KnowledgeBase | None:
        """更新知识库

        Args:
            kb_id: 知识库 ID
            data: 更新请求
            tenant_id: 租户 ID

        Returns:
            更新后的知识库
        """
        update_data: dict = data.model_dump(exclude_unset=True, exclude_none=True)
        if data.chunking_config:
            update_data["chunking_config"] = data.chunking_config.model_dump()

        return await self.kb_repo.update(kb_id, **update_data)

    async def delete_knowledge_base(
        self, kb_id: str, tenant_id: int
    ) -> bool:
        """删除知识库（软删除）

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        return await self.kb_repo.soft_delete(kb_id, tenant_id)

    async def hybrid_search(
        self, kb_id: str, request: HybridSearchRequest, tenant_id: int
    ) -> list[dict]:
        """混合搜索（向量 + 关键词 + RRF 融合 + 重排序）

        对齐 WeKnora99 的 HybridSearch 功能

        Args:
            kb_id: 知识库 ID
            request: 搜索请求
            tenant_id: 租户 ID

        Returns:
            搜索结果列表
        """
        kb = await self.get_knowledge_base(kb_id, tenant_id)
        if not kb:
            logger.warning(
                "hybrid_search_kb_not_found",
                kb_id=kb_id,
                tenant_id=tenant_id,
            )
            return []

        logger.info(
            "hybrid_search_start",
            kb_id=kb_id,
            query=request.query_text,
            match_count=request.match_count,
            vector_threshold=request.vector_threshold,
            keyword_threshold=request.keyword_threshold,
            enable_rerank=request.enable_rerank,
        )

        # 构建搜索参数
        search_params = SearchParams(
            query_text=request.query_text,
            vector_threshold=request.vector_threshold,
            keyword_threshold=request.keyword_threshold,
            match_count=request.match_count,
            disable_keywords_match=request.disable_keywords_match,
            disable_vector_match=request.disable_vector_match,
            knowledge_ids=request.knowledge_ids,
            tag_ids=request.tag_ids,
            only_recommended=request.only_recommended,
            enable_rerank=request.enable_rerank,
            rerank_model_id=request.rerank_model_id or kb.rerank_model_id,
            top_k=request.top_k,
        )

        # 创建混合搜索服务
        search_service = HybridSearchService(
            session=self.session,
            embedding_model_id=kb.embedding_model_id,
        )

        # 执行搜索
        try:
            search_results = await search_service.search(
                kb_id=kb_id,
                tenant_id=tenant_id,
                params=search_params,
            )

            # 转换为字典格式
            results = [
                {
                    "content": r.content,
                    "score": r.score,
                    "chunk_id": r.id,
                    "knowledge_id": r.knowledge_id,
                    "knowledge_title": r.knowledge_title,
                    "metadata": r.metadata,
                    "chunk_index": r.chunk_index,
                    "start_at": r.start_at,
                    "end_at": r.end_at,
                    "match_type": r.match_type,
                    "chunk_type": r.chunk_type,
                    "parent_chunk_id": r.parent_chunk_id,
                    "image_info": r.image_info,
                    "knowledge_filename": r.knowledge_filename,
                    "knowledge_source": r.knowledge_source,
                    "chunk_metadata": r.chunk_metadata,
                    "matched_content": r.matched_content,
                }
                for r in search_results
            ]

            logger.info(
                "hybrid_search_complete",
                kb_id=kb_id,
                query=request.query_text,
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "hybrid_search_error",
                kb_id=kb_id,
                query=request.query_text,
                error=str(e),
            )
            return []

    async def get_knowledge_count(self, kb_id: str, tenant_id: int) -> int:
        """获取知识库下的知识数量

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID

        Returns:
            知识条目数量
        """
        return await self.kb_repo.get_knowledge_count(kb_id, tenant_id)


class KnowledgeService:
    """知识条目服务"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._knowledge_repo: KnowledgeRepository | None = None
        self._chunk_repo: ChunkRepository | None = None

    @property
    def knowledge_repo(self) -> KnowledgeRepository:
        """延迟初始化知识条目仓储"""
        if self._knowledge_repo is None:
            self._knowledge_repo = KnowledgeRepository(self.session)
        return self._knowledge_repo

    @property
    def chunk_repo(self) -> ChunkRepository:
        """延迟初始化分块仓储"""
        if self._chunk_repo is None:
            self._chunk_repo = ChunkRepository(self.session)
        return self._chunk_repo

    async def create_from_file(
        self,
        kb_id: str,
        file_path: str,
        file_name: str,
        file_type: str,
        file_size: int,
        tenant_id: int,
        enable_multimodel: bool = True,
    ) -> dict:
        """从文件创建知识条目

        Args:
            kb_id: 知识库 ID
            file_path: 文件路径
            file_name: 文件名
            file_type: 文件类型
            file_size: 文件大小
            tenant_id: 租户 ID
            enable_multimodel: 是否启用多模态

        Returns:
            创建结果
        """
        # 1. 获取知识库配置
        kb = await KnowledgeBaseService(self.session).get_knowledge_base(
            kb_id, tenant_id
        )
        if not kb:
            raise ValueError(f"Knowledge base {kb_id} not found")

        # 2. 创建知识条目记录
        knowledge = await self.knowledge_repo.create_from_file(
            kb_id=kb_id,
            tenant_id=tenant_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            title=file_name,
        )

        logger.info(
            "knowledge_created_from_file",
            knowledge_id=knowledge.id,
            kb_id=kb_id,
            file_name=file_name,
        )

        # 3. 处理文档内容（分块）
        try:
            chunks_data = await self._process_file(
                file_path=file_path,
                kb=kb,
                knowledge_id=knowledge.id,
            )

            # 4. 保存分块
            chunks = await self.chunk_repo.create_chunks(
                chunks=chunks_data,
                kb_id=kb_id,
                knowledge_id=knowledge.id,
                tenant_id=tenant_id,
            )

            # 5. 更新分块数量和解析状态
            await self.knowledge_repo.update_chunk_count(knowledge.id, len(chunks))
            await self.knowledge_repo.update_parse_status(knowledge.id, "completed")

            logger.info(
                "knowledge_file_processed",
                knowledge_id=knowledge.id,
                chunk_count=len(chunks),
            )

            return {
                "id": knowledge.id,
                "status": "completed",
                "message": "文档处理完成",
                "chunk_count": len(chunks),
            }

        except Exception as e:
            logger.error(
                "knowledge_file_process_failed",
                knowledge_id=knowledge.id,
                error=str(e),
            )
            await self.knowledge_repo.update_parse_status(
                knowledge.id, "failed", str(e)
            )
            return {
                "id": knowledge.id,
                "status": "failed",
                "message": f"文档处理失败: {str(e)}",
            }

    async def _process_file(
        self,
        file_path: str,
        kb: KnowledgeBase,
        knowledge_id: str,
    ) -> list[dict]:
        """处理文件内容，返回分块数据

        Args:
            file_path: 文件路径
            kb: 知识库配置
            knowledge_id: 知识条目 ID

        Returns:
            分块数据列表
        """
        # 检查文件类型
        if not is_supported_file_type(file_path):
            raise ValueError(f"Unsupported file type: {get_file_extension(file_path)}")

        # 获取文档加载器
        loader_class = get_file_loader(file_path)
        if not loader_class:
            raise ValueError(f"No loader available for: {file_path}")

        # 加载文档
        loader = loader_class(file_path, encoding="utf-8")
        documents = loader.load()

        logger.info(
            "file_documents_loaded",
            file_path=file_path,
            document_count=len(documents),
        )

        # 获取分块配置
        chunking_config = kb.chunking_config or {}
        chunk_size = chunking_config.get("chunk_size", 1000)
        chunk_overlap = chunking_config.get("chunk_overlap", 200)
        separators = chunking_config.get("separators", ["\n\n", "\n", "。", ".", " ", ""])

        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        # 分割文档
        chunks = []
        current_index = 0

        for doc in documents:
            split_docs = text_splitter.split_documents([doc])
            for i, split_doc in enumerate(split_docs):
                chunks.append(
                    {
                        "content": split_doc.page_content,
                        "chunk_index": current_index + i,
                        "start_at": 0,  # LangChain 不提供位置信息
                        "end_at": len(split_doc.page_content),
                        "is_enabled": True,
                        "chunk_type": "text",
                        "meta_data": {
                            "source": split_doc.metadata.get("source", ""),
                            "page": split_doc.metadata.get("page", 0),
                        },
                    }
                )
            current_index += len(split_docs)

        return chunks

    async def create_from_url(
        self,
        kb_id: str,
        url: str,
        tenant_id: int,
        enable_multimodel: bool = True,
    ) -> dict:
        """从 URL 创建知识条目

        Args:
            kb_id: 知识库 ID
            url: URL 地址
            tenant_id: 租户 ID
            enable_multimodel: 是否启用多模态

        Returns:
            创建结果
        """
        # 1. 先创建知识条目记录
        knowledge = await self.knowledge_repo.create_from_url(
            kb_id=kb_id,
            tenant_id=tenant_id,
            url=url,
        )

        # 2. 获取知识库配置（用于分块）
        kb = await KnowledgeBaseService(self.session).get_knowledge_base(
            kb_id, tenant_id
        )
        if not kb:
            await self.knowledge_repo.update_parse_status(
                knowledge.id, "failed", "Knowledge base not found"
            )
            return {
                "id": knowledge.id,
                "status": "failed",
                "message": "Knowledge base not found",
            }

        # 3. 抓取 URL 内容
        content = ""
        try:
            crawler = JinaReader(CrawlerConfig(max_content_length=50000))
            content = await crawler.crawl(url, return_format="markdown")
        except Exception as e:
            logger.error("url_crawl_failed", url=url, error=str(e))
            await self.knowledge_repo.update_parse_status(
                knowledge.id, "failed", str(e)
            )
            return {
                "id": knowledge.id,
                "status": "failed",
                "message": f"URL 抓取失败: {str(e)}",
            }

        if not content or content.startswith("Error:"):
            await self.knowledge_repo.update_parse_status(
                knowledge.id, "failed", content[:200] if content else "Empty content"
            )
            return {
                "id": knowledge.id,
                "status": "failed",
                "message": "URL 内容为空或抓取失败",
            }

        # 4. 分块处理
        chunking_config = kb.chunking_config or {}
        chunk_size = chunking_config.get("chunk_size", 1000)
        chunk_overlap = chunking_config.get("chunk_overlap", 200)
        separators = chunking_config.get("separators", ["\n\n", "\n", "。", ".", " ", ""])

        chunks_data = self._split_text(
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        # 5. 保存分块
        chunks = await self.chunk_repo.create_chunks(
            chunks=chunks_data,
            kb_id=kb_id,
            knowledge_id=knowledge.id,
            tenant_id=tenant_id,
        )

        # 6. 更新分块数量和解析状态
        await self.knowledge_repo.update_chunk_count(knowledge.id, len(chunks))
        await self.knowledge_repo.update_parse_status(knowledge.id, "completed")

        logger.info(
            "knowledge_created_from_url",
            knowledge_id=knowledge.id,
            kb_id=kb_id,
            url=url,
            chunk_count=len(chunks),
        )

        return {
            "id": knowledge.id,
            "status": "completed",
            "message": "URL 内容已成功抓取并分块",
            "chunk_count": len(chunks),
        }

    def _split_text(
        self,
        content: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> list[dict]:
        """将文本分割成块

        Args:
            content: 文本内容
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            separators: 分隔符列表

        Returns:
            分块数据列表
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", ".", " ", ""]

        chunks = []
        start = 0
        content_length = len(content)

        while start < content_length:
            end = start + chunk_size

            # 尝试在分隔符处分割
            if end < content_length:
                for sep in separators:
                    sep_pos = content.rfind(sep, start, end)
                    if sep_pos != -1:
                        end = sep_pos + len(sep)
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(
                    {
                        "content": chunk_content,
                        "chunk_index": len(chunks),
                        "start_at": start,
                        "end_at": end,
                        "is_enabled": True,
                        "chunk_type": "text",
                    }
                )

            # 计算下一个块的起始位置（带重叠）
            start = end - chunk_overlap if end < content_length else end

        return chunks

    async def get_knowledge(
        self, knowledge_id: str, tenant_id: int
    ) -> dict | None:
        """获取知识条目详情

        Args:
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            知识条目详情
        """
        knowledge = await self.knowledge_repo.get_by_tenant(knowledge_id, tenant_id)
        if not knowledge:
            return None

        return {
            "id": knowledge.id,
            "knowledge_base_id": knowledge.knowledge_base_id,
            "type": knowledge.type,
            "title": knowledge.title,
            "source": knowledge.source,
            "parse_status": knowledge.parse_status,
            "enable_status": knowledge.enable_status,
            "file_name": knowledge.file_name,
            "file_size": knowledge.file_size,
            "chunk_count": (
                knowledge.meta_data.get("chunk_count", 0) if knowledge.meta_data else 0
            ),
            "created_at": knowledge.created_at,
            "processed_at": knowledge.processed_at,
        }

    async def list_knowledge(
        self, kb_id: str, tenant_id: int, params: PaginationParams
    ) -> list[dict]:
        """知识条目列表

        Args:
            kb_id: 知识库 ID
            tenant_id: 租户 ID
            params: 分页参数

        Returns:
            知识条目列表
        """
        result = await self.knowledge_repo.list_by_kb(kb_id, tenant_id, params)

        return [
            {
                "id": k.id,
                "knowledge_base_id": k.knowledge_base_id,
                "type": k.type,
                "title": k.title,
                "source": k.source,
                "parse_status": k.parse_status,
                "enable_status": k.enable_status,
                "file_name": k.file_name,
                "file_size": k.file_size,
                "chunk_count": (
                    k.meta_data.get("chunk_count", 0) if k.meta_data else 0
                ),
                "created_at": k.created_at,
                "processed_at": k.processed_at,
            }
            for k in result.items
        ]

    async def delete_knowledge(
        self, knowledge_id: str, tenant_id: int
    ) -> bool:
        """删除知识条目

        Args:
            knowledge_id: 知识条目 ID
            tenant_id: 租户 ID

        Returns:
            是否删除成功
        """
        # 先删除关联的分块
        await self.chunk_repo.delete_by_knowledge(knowledge_id, tenant_id)

        return await self.knowledge_repo.soft_delete(knowledge_id, tenant_id)

    async def update_knowledge(
        self, knowledge_id: str, data: KnowledgeUpdate, tenant_id: int
    ) -> dict | None:
        """更新知识条目

        Args:
            knowledge_id: 知识条目 ID
            data: 更新请求
            tenant_id: 租户 ID

        Returns:
            更新后的知识条目详情
        """
        update_data: dict = {}
        if data.title is not None:
            update_data["title"] = data.title
        if data.enable_status is not None:
            update_data["enable_status"] = data.enable_status

        if not update_data:
            return await self.get_knowledge(knowledge_id, tenant_id)

        knowledge = await self.knowledge_repo.update(
            knowledge_id, **update_data
        )

        if not knowledge:
            return None

        logger.info(
            "knowledge_updated",
            knowledge_id=knowledge_id,
            update_data=update_data,
        )

        return {
            "id": knowledge.id,
            "knowledge_base_id": knowledge.knowledge_base_id,
            "type": knowledge.type,
            "title": knowledge.title,
            "source": knowledge.source,
            "parse_status": knowledge.parse_status,
            "enable_status": knowledge.enable_status,
            "file_name": knowledge.file_name,
            "file_size": knowledge.file_size,
            "chunk_count": (
                knowledge.meta_data.get("chunk_count", 0) if knowledge.meta_data else 0
            ),
            "created_at": knowledge.created_at,
            "processed_at": knowledge.processed_at,
        }

    async def create_manual(
        self,
        kb_id: str,
        title: str,
        content: str,
        tenant_id: int,
    ) -> dict:
        """手工创建知识条目

        对齐 WeKnora99 POST /knowledge-bases/{id}/knowledge/manual

        Args:
            kb_id: 知识库 ID
            title: 知识条目标题
            content: Markdown 内容
            tenant_id: 租户 ID

        Returns:
            创建结果
        """
        # 1. 获取知识库配置
        kb = await KnowledgeBaseService(self.session).get_knowledge_base(
            kb_id, tenant_id
        )
        if not kb:
            raise ValueError(f"Knowledge base {kb_id} not found")

        # 2. 创建知识条目记录
        knowledge = await self.knowledge_repo.create_manual(
            kb_id=kb_id,
            tenant_id=tenant_id,
            title=title,
            content=content,
        )

        logger.info(
            "manual_knowledge_created",
            knowledge_id=knowledge.id,
            kb_id=kb_id,
            title=title,
        )

        # 3. 分块处理
        chunking_config = kb.chunking_config or {}
        chunk_size = chunking_config.get("chunk_size", 1000)
        chunk_overlap = chunking_config.get("chunk_overlap", 200)
        separators = chunking_config.get("separators", ["\n\n", "\n", "。", ".", " ", ""])

        chunks_data = self._split_text(
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        # 4. 保存分块
        chunks = await self.chunk_repo.create_chunks(
            chunks=chunks_data,
            kb_id=kb_id,
            knowledge_id=knowledge.id,
            tenant_id=tenant_id,
        )

        # 5. 更新分块数量和解析状态
        await self.knowledge_repo.update_chunk_count(knowledge.id, len(chunks))
        await self.knowledge_repo.update_parse_status(knowledge.id, "completed")

        return {
            "id": knowledge.id,
            "status": "completed",
            "message": "手工知识创建成功",
            "chunk_count": len(chunks),
        }


__all__ = [
    "KnowledgeBaseService",
    "KnowledgeService",
]
