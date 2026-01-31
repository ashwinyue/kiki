"""对话式检索器

使用 LangChain 的 ConversationalRetrievalChain 实现对话式检索。
支持聊天历史上下文，可结合不同检索器使用。

依赖安装:
    uv add langchain langchain-community

使用示例:
```python
from app.retrievers import ConversationalRetriever, ConversationalRetrieverConfig
from app.llm import get_embeddings
from app.vector_stores import create_vector_store, VectorStoreConfig

# 创建向量存储
embeddings = get_embeddings(provider="dashscope")
vector_store = create_vector_store("memory", VectorStoreConfig(), embeddings)
retriever = await vector_store.as_retriever(k=5)

# 创建对话式检索器
config = ConversationalRetrieverConfig()
conv_retriever = ConversationalRetriever(
    config=config,
    retriever=retriever,
)

# 执行查询
history = []  # 聊天历史
answer = await conv_retriever.asearch(
    query="如何使用 Python 进行异步编程？",
    chat_history=history,
)
```
"""

from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationalRetrieverConfig:
    """对话式检索器配置

    Attributes:
        k: 返回结果数量
        search_type: 搜索类型 (similarity/score_threshold/mmr)
        score_threshold: 相似度阈值
        condense_question_prompt: 问题重述提示模板
        return_source_documents: 是否返回源文档
        max_tokens_limit: 最大 token 限制
        output_language: 输出语言
        tenant_id: 租户 ID
    """

    # 检索配置
    k: int = 5
    search_type: Literal["similarity", "score_threshold", "mmr"] = "similarity"
    score_threshold: float | None = None

    # 对话配置
    return_source_documents: bool = True
    max_tokens_limit: int = 3000
    output_language: str = "zh"

    # 问题重述配置
    condense_question_prompt: str = (
        "Given the following conversation and a follow up question, "
        "rephrase the follow up question to be a standalone question. "
        "Chat History:\\n{chat_history}\\n"
        "Follow Up Input: {question}\\n"
        "Standalone question:"
    )

    # 回答生成配置
    qa_prompt: str = (
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, "
        "don't try to make up an answer.\\n"
        "Context:\\n{context}\\n"
        "Question:\\n{question}\\n"
        "Helpful Answer:"
    )

    tenant_id: int | None = None


@dataclass
class SearchResult:
    """搜索结果

    Attributes:
        answer: 答案
        source_documents: 源文档列表
        question: 重述后的问题
    """

    answer: str
    source_documents: list[Document]
    question: str


class ConversationalRetriever:
    """对话式检索器

    支持基于聊天历史的检索和问答。
    """

    config: ConversationalRetrieverConfig
    _retriever: BaseRetriever
    _llm: BaseLanguageModel | None

    def __init__(
        self,
        config: ConversationalRetrieverConfig,
        retriever: BaseRetriever,
        llm: BaseLanguageModel | None = None,
    ):
        """初始化对话式检索器

        Args:
            config: 检索器配置
            retriever: 底层检索器
            llm: LLM 实例（用于生成答案和重述问题）
        """
        self.config = config
        self._retriever = retriever
        self._llm = llm

    async def asearch(
        self,
        query: str,
        chat_history: list[tuple[str, str]] | list[BaseMessage] | None = None,
    ) -> SearchResult:
        """异步搜索并生成答案

        Args:
            query: 查询文本
            chat_history: 聊天历史，格式为 [(question, answer), ...] 或消息列表

        Returns:
            搜索结果
        """
        if chat_history is None:
            chat_history = []

        # 1. 重述问题（如果需要）
        standalone_question = query
        if chat_history and self._llm:
            standalone_question = await self._condense_question(
                question=query,
                chat_history=chat_history,
            )

        # 2. 检索相关文档
        documents = await self._retrieve_documents(standalone_question)

        # 3. 生成答案
        answer = await self._generate_answer(
            question=standalone_question,
            documents=documents,
            chat_history=chat_history,
        )

        logger.info(
            "conversational_search_completed",
            query=query[:100],
            standalone_question=standalone_question[:100],
            document_count=len(documents),
            answer_length=len(answer),
        )

        return SearchResult(
            answer=answer,
            source_documents=documents,
            question=standalone_question,
        )

    async def aretrieve(
        self,
        query: str,
        chat_history: list[tuple[str, str]] | list[BaseMessage] | None = None,
    ) -> list[Document]:
        """仅检索文档，不生成答案

        Args:
            query: 查询文本
            chat_history: 聊天历史

        Returns:
            检索到的文档列表
        """
        if chat_history is None:
            chat_history = []

        # 重述问题
        standalone_question = query
        if chat_history and self._llm:
            standalone_question = await self._condense_question(
                question=query,
                chat_history=chat_history,
            )

        return await self._retrieve_documents(standalone_question)

    async def _condense_question(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | list[BaseMessage],
    ) -> str:
        """重述问题

        基于聊天历史将当前问题重述为独立问题。

        Args:
            question: 当前问题
            chat_history: 聊天历史

        Returns:
            重述后的问题
        """
        if not self._llm:
            return question

        # 转换历史格式
        history_str = self._format_chat_history(chat_history)

        # 构建提示
        prompt = PromptTemplate.from_template(self.config.condense_question_prompt)

        # 生成重述问题
        try:
            result = await self._llm.apredict(
                prompt.format(
                    chat_history=history_str,
                    question=question,
                )
            )
            standalone_question = result.strip()

            logger.debug(
                "question_condensed",
                original=question[:100],
                condensed=standalone_question[:100],
            )

            return standalone_question

        except Exception as e:
            logger.error(
                "condense_question_failed",
                error=str(e),
            )
            return question

    async def _retrieve_documents(self, query: str) -> list[Document]:
        """检索文档

        Args:
            query: 查询文本

        Returns:
            文档列表
        """
        try:
            if hasattr(self._retriever, "aretrieve"):
                docs = await self._retriever.aretrieve(query)
            else:
                docs = self._retriever.get_relevant_documents(query)

            # 应用 k 限制
            docs = docs[: self.config.k]

            # 应用阈值过滤
            if self.config.score_threshold is not None:
                docs = [
                    d
                    for d in docs
                    if d.metadata.get("score", 1.0) >= self.config.score_threshold
                ]

            return docs

        except Exception as e:
            logger.error(
                "retrieve_documents_failed",
                query=query[:100],
                error=str(e),
            )
            return []

    async def _generate_answer(
        self,
        question: str,
        documents: list[Document],
        chat_history: list[tuple[str, str]] | list[BaseMessage],
    ) -> str:
        """生成答案

        Args:
            question: 问题
            documents: 检索到的文档
            chat_history: 聊天历史

        Returns:
            生成的答案
        """
        if not self._llm:
            # 没有 LLM 时，返回文档摘要
            return self._summarize_documents(documents)

        # 构建上下文
        context = self._format_context(documents)

        # 构建提示
        prompt = PromptTemplate.from_template(self.config.qa_prompt)

        # 生成答案
        try:
            answer = await self._llm.apredict(
                prompt.format(
                    context=context,
                    question=question,
                )
            )

            return answer.strip()

        except Exception as e:
            logger.error(
                "generate_answer_failed",
                error=str(e),
            )
            return self._summarize_documents(documents)

    def _format_chat_history(
        self,
        chat_history: list[tuple[str, str]] | list[BaseMessage],
    ) -> str:
        """格式化聊天历史

        Args:
            chat_history: 聊天历史

        Returns:
            格式化后的历史字符串
        """
        if not chat_history:
            return ""

        lines = []
        for item in chat_history:
            if isinstance(item, tuple):
                question, answer = item
                lines.append(f"Human: {question}")
                lines.append(f"Assistant: {answer}")
            else:
                # BaseMessage
                lines.append(f"{item.type}: {item.content}")

        return "\\n".join(lines)

    def _format_context(self, documents: list[Document]) -> str:
        """格式化文档上下文

        Args:
            documents: 文档列表

        Returns:
            格式化后的上下文字符串
        """
        if not documents:
            return "No relevant context found."

        # 控制 token 数量
        total_chars = 0
        max_chars = self.config.max_tokens_limit * 3  # 粗略估算

        context_parts = []
        for doc in documents:
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                # 截断
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(content[:remaining] + "...")
                break
            context_parts.append(content)
            total_chars += len(content)

        return "\\n\\n".join(context_parts)

    def _summarize_documents(self, documents: list[Document]) -> str:
        """摘要文档（无 LLM 时使用）

        Args:
            documents: 文档列表

        Returns:
            文档摘要
        """
        if not documents:
            return "抱歉，没有找到相关信息。"

        summary_parts = []
        for i, doc in enumerate(documents[:3], 1):  # 最多返回3条
            summary_parts.append(f"{i}. {doc.page_content[:200]}...")

        answer = "根据知识库找到以下相关信息：\\n\\n"
        answer += "\\n".join(summary_parts)

        return answer

    def set_llm(self, llm: BaseLanguageModel) -> None:
        """设置 LLM

        Args:
            llm: LLM 实例
        """
        self._llm = llm
        logger.info("conversational_retriever_llm_set")

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """设置底层检索器

        Args:
            retriever: 检索器实例
        """
        self._retriever = retriever
        logger.info("conversational_retriever_retriever_set")

    def get_chat_history_string(
        self,
        chat_history: list[tuple[str, str]] | list[BaseMessage],
    ) -> str:
        """获取格式化的聊天历史字符串

        Args:
            chat_history: 聊天历史

        Returns:
            格式化的历史字符串
        """
        return self._format_chat_history(chat_history)


__all__ = [
    "ConversationalRetriever",
    "ConversationalRetrieverConfig",
    "SearchResult",
]
