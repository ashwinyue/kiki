"""数据集服务

对齐 WeKnora99 DatasetService。

支持从 Parquet 文件或数据库加载评估数据集。

数据集格式:
- queries: 查询列表 (id -> question)
- corpus: 文档列表 (id -> passage)
- answers: 答案列表 (id -> answer)
- qrels: 查询-文档相关性 (query_id -> [doc_id, ...])
- qas: 查询-答案对应 (query_id -> answer_id)
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from app.evaluation.types import QAPair
from app.observability.logging import get_logger

logger = get_logger(__name__)


# 默认数据集目录
DEFAULT_DATASET_DIR = "dataset/samples"


@dataclass
class DatasetConfig:
    """数据集配置"""

    dir_path: str = DEFAULT_DATASET_DIR
    max_samples: int | None = None


class DatasetService:
    """数据集服务

    提供数据集加载和管理功能。
    """

    def __init__(self, config: DatasetConfig | None = None):
        """初始化数据集服务

        Args:
            config: 数据集配置
        """
        self._config = config or DatasetConfig()
        self._datasets: dict[str, Dataset] = {}

    async def get_dataset(
        self,
        dataset_name: str,
    ) -> list[QAPair]:
        """获取数据集

        Args:
            dataset_name: 数据集名称

        Returns:
            QA 对列表

        Raises:
            ValueError: 数据集不存在
        """
        logger.info(
            "dataset_load_start",
            dataset=dataset_name,
        )

        # 检查缓存
        if dataset_name in self._datasets:
            dataset = self._datasets[dataset_name]
            qa_pairs = dataset.iterate()
            logger.info(
                "dataset_load_from_cache",
                dataset=dataset_name,
                count=len(qa_pairs),
            )
            return qa_pairs

        # 加载数据集
        dataset = await self._load_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"数据集不存在: {dataset_name}")

        self._datasets[dataset_name] = dataset
        qa_pairs = dataset.iterate()

        logger.info(
            "dataset_load_success",
            dataset=dataset_name,
            count=len(qa_pairs),
        )

        return qa_pairs

    async def _load_dataset(self, dataset_name: str) -> "Dataset | None":
        """加载数据集

        Args:
            dataset_name: 数据集名称

        Returns:
            Dataset 实例或 None
        """
        # 尝试从目录加载
        dir_path = self._config.dir_path

        # 支持多种数据格式
        for format in ["json", "parquet"]:
            try:
                if format == "json":
                    dataset = await self._load_from_json(dir_path, dataset_name)
                else:
                    dataset = await self._load_from_parquet(dir_path, dataset_name)

                if dataset is not None:
                    return dataset
            except Exception as e:
                logger.debug(
                    "dataset_load_format_failed",
                    format=format,
                    dataset=dataset_name,
                    error=str(e),
                )

        # 尝试加载示例数据集
        return await self._load_sample_dataset(dataset_name)

    async def _load_from_json(
        self,
        dir_path: str,
        dataset_name: str,
    ) -> "Dataset | None":
        """从 JSON 文件加载数据集

        Args:
            dir_path: 数据目录
            dataset_name: 数据集名称

        Returns:
            Dataset 实例或 None
        """
        json_file = os.path.join(dir_path, f"{dataset_name}.json")
        if not os.path.exists(json_file):
            return None

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Dataset.from_dict(data)

    async def _load_from_parquet(
        self,
        dir_path: str,
        dataset_name: str,
    ) -> "Dataset | None":
        """从 Parquet 文件加载数据集

        Args:
            dir_path: 数据目录
            dataset_name: 数据集名称

        Returns:
            Dataset 实例或 None
        """
        try:
            import pandas as pd

            queries_file = os.path.join(dir_path, f"{dataset_name}/queries.parquet")
            corpus_file = os.path.join(dir_path, f"{dataset_name}/corpus.parquet")
            answers_file = os.path.join(dir_path, f"{dataset_name}/answers.parquet")
            qrels_file = os.path.join(dir_path, f"{dataset_name}/qrels.parquet")
            qas_file = os.path.join(dir_path, f"{dataset_name}/qas.parquet")

            if not all(os.path.exists(f) for f in [queries_file, corpus_file]):
                return None

            # 加载 Parquet 文件
            queries_df = pd.read_parquet(queries_file)
            corpus_df = pd.read_parquet(corpus_file)

            answers_df = None
            if os.path.exists(answers_file):
                answers_df = pd.read_parquet(answers_file)

            qrels_df = None
            if os.path.exists(qrels_file):
                qrels_df = pd.read_parquet(qrels_file)

            qas_df = None
            if os.path.exists(qas_file):
                qas_df = pd.read_parquet(qas_file)

            return Dataset.from_parquet(
                queries_df=queries_df,
                corpus_df=corpus_df,
                answers_df=answers_df,
                qrels_df=qrels_df,
                qas_df=qas_df,
            )

        except ImportError:
            logger.warning("pandas_parquet_not_available")
            return None

    async def _load_sample_dataset(self, dataset_name: str) -> "Dataset | None":
        """加载示例数据集

        Args:
            dataset_name: 数据集名称

        Returns:
            Dataset 实例或 None
        """
        # 如果请求的是默认数据集，创建示例数据
        if dataset_name in ["default", "sample"]:
            return Dataset.create_sample()

        return None

    def clear_cache(self, dataset_name: str | None = None) -> None:
        """清除数据集缓存

        Args:
            dataset_name: 指定数据集名称，None 表示清除所有
        """
        if dataset_name is None:
            self._datasets.clear()
            logger.info("dataset_cache_cleared_all")
        elif dataset_name in self._datasets:
            del self._datasets[dataset_name]
            logger.info("dataset_cache_cleared", dataset=dataset_name)


class Dataset:
    """数据集

    存储查询、文档、答案和相关性标注。
    """

    def __init__(
        self,
        queries: dict[int, str],
        corpus: dict[int, str],
        answers: dict[int, str],
        qrels: dict[int, list[int]],
        qas: dict[int, int],
    ):
        """初始化数据集

        Args:
            queries: 查询字典 (id -> question)
            corpus: 文档字典 (id -> passage)
            answers: 答案字典 (id -> answer)
            qrels: 相关性标注 (query_id -> [doc_id, ...])
            qas: 查询-答案对应 (query_id -> answer_id)
        """
        self.queries = queries
        self.corpus = corpus
        self.answers = answers
        self.qrels = qrels
        self.qas = qas

    def iterate(self) -> list[QAPair]:
        """迭代生成 QA 对

        Returns:
            QA 对列表
        """
        pairs = []

        for qid, question in self.queries.items():
            # 获取答案
            answer = ""
            if qid in self.qas:
                aid = self.qas[qid]
                if aid in self.answers:
                    answer = self.answers[aid]

            # 获取相关文档
            pids = self.qrels.get(qid, [])
            passages = [self.corpus.get(pid, "") for pid in pids if pid in self.corpus]

            pair = QAPair(
                qid=qid,
                question=question,
                pids=pids,
                passages=passages,
                aid=self.qas.get(qid, 0),
                answer=answer,
            )
            pairs.append(pair)

        return pairs

    def get_context_for_qid(self, qid: int) -> list[str]:
        """获取查询的上下文文档

        Args:
            qid: 查询 ID

        Returns:
            上下文文档列表
        """
        pids = self.qrels.get(qid, [])
        return [self.corpus.get(pid, "") for pid in pids if pid in self.corpus]

    def stats(self) -> dict[str, Any]:
        """获取数据集统计信息

        Returns:
            统计信息字典
        """
        total_relations = sum(len(pids) for pids in self.qrels.values())
        avg_passages = total_relations / len(self.qrels) if self.qrels else 0

        covered_queries = len(self.qas)
        coverage = covered_queries / len(self.queries) * 100 if self.queries else 0

        return {
            "total_queries": len(self.queries),
            "total_corpus": len(self.corpus),
            "total_answers": len(self.answers),
            "total_relations": total_relations,
            "avg_passages_per_query": round(avg_passages, 2),
            "answer_coverage": round(coverage, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "queries": {str(k): v for k, v in self.queries.items()},
            "corpus": {str(k): v for k, v in self.corpus.items()},
            "answers": {str(k): v for k, v in self.answers.items()},
            "qrels": {str(k): v for k, v in self.qrels.items()},
            "qas": {str(k): v for k, v in self.qas.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dataset":
        """从字典创建数据集"""
        return cls(
            queries={int(k): v for k, v in data.get("queries", {}).items()},
            corpus={int(k): v for k, v in data.get("corpus", {}).items()},
            answers={int(k): v for k, v in data.get("answers", {}).items()},
            qrels={int(k): v for k, v in data.get("qrels", {}).items()},
            qas={int(k): v for k, v in data.get("qas", {}).items()},
        )

    @classmethod
    def from_parquet(
        cls,
        queries_df: Any,
        corpus_df: Any,
        answers_df: Any | None = None,
        qrels_df: Any | None = None,
        qas_df: Any | None = None,
    ) -> "Dataset":
        """从 Parquet DataFrame 创建数据集"""
        # 提取查询
        queries = {}
        for _, row in queries_df.iterrows():
            queries[int(row["id"])] = row["text"]

        # 提取文档
        corpus = {}
        for _, row in corpus_df.iterrows():
            corpus[int(row["id"])] = row["text"]

        # 提取答案
        answers = {}
        if answers_df is not None:
            for _, row in answers_df.iterrows():
                answers[int(row["id"])] = row["text"]

        # 提取相关性标注
        qrels = {}
        if qrels_df is not None:
            for _, row in qrels_df.iterrows():
                qid = int(row["qid"])
                pid = int(row["pid"])
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(pid)

        # 提取 QA 对应
        qas = {}
        if qas_df is not None:
            for _, row in qas_df.iterrows():
                qas[int(row["qid"])] = int(row["aid"])

        return cls(
            queries=queries,
            corpus=corpus,
            answers=answers,
            qrels=qrels,
            qas=qas,
        )

    @classmethod
    def create_sample(cls) -> "Dataset":
        """创建示例数据集

        Returns:
            示例数据集
        """
        # 示例查询
        queries = {
            1: "什么是 Python 编程语言？",
            2: "如何安装 Python？",
            3: "Python 有哪些主要特性？",
            4: "什么是虚拟环境？",
            5: "如何创建虚拟环境？",
        }

        # 示例文档
        corpus = {
            1: "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
            2: "Python 支持多种编程范式，包括面向对象、函数式和过程式编程。",
            3: "Python 的设计哲学强调代码可读性，使用缩进定义代码块。",
            4: "Python 是一种解释型语言，代码在运行时逐行解释执行。",
            5: "Python 拥有丰富的标准库和第三方包生态系统。",
            6: "安装 Python 可以从官网下载安装包，或使用包管理器如 apt、brew。",
            7: "Python 3.x 是当前的主要版本，Python 2.x 已停止支持。",
            8: "虚拟环境是 Python 的独立运行环境，可以隔离不同项目的依赖。",
            9: "可以使用 venv 模块创建虚拟环境：python -m venv myenv。",
            10: "pip 是 Python 的包管理工具，用于安装和管理第三方包。",
        }

        # 示例答案
        answers = {
            1: "Python 是一种高级解释型编程语言。",
            2: "可以从 python.org 下载或使用包管理器安装。",
            3: "Python 具有简单易学、可读性强、生态丰富等特点。",
            4: "虚拟环境是隔离的 Python 运行环境。",
            5: "使用 python -m venv 命令创建虚拟环境。",
        }

        # 相关性标注
        qrels = {
            1: [1, 2, 3, 4, 5],
            2: [6, 7],
            3: [1, 2, 3, 5],
            4: [8],
            5: [8, 9],
        }

        # QA 对应
        qas = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        return cls(
            queries=queries,
            corpus=corpus,
            answers=answers,
            qrels=qrels,
            qas=qas,
        )


# 全局数据集服务实例
_dataset_service: DatasetService | None = None


def get_dataset_service() -> DatasetService:
    """获取数据集服务实例"""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service


__all__ = [
    "DatasetService",
    "Dataset",
    "DatasetConfig",
    "get_dataset_service",
]
