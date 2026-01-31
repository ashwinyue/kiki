"""ROUGE 评估指标

对齐 WeKnora99 ROUGE 实现。
"""

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel


class RougeMetric(BaseModel):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 指标

    用于评估生成文本与参考文本的相似度。
    广泛应用于文本摘要任务。
    """

    variant: str = "rouge-1"  # rouge-1, rouge-2, rouge-l
    stat: str = "f"  # f, p, r

    @property
    def name(self) -> str:
        return f"{self.variant}"

    def _tokenize(self, text: str) -> list[str]:
        """简单分词"""
        return re.findall(r"\w+", text.lower())

    def _get_ngrams(self, text: str, n: int) -> Counter:
        """获取 n-gram 计数"""
        words = self._tokenize(text)
        if len(words) < n:
            return Counter()
        return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    def _lcs_length(self, words1: list[str], words2: list[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(words1), len(words2)
        if m == 0 or n == 0:
            return 0

        # 动态规划计算 LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _compute_rouge_n(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
        n: int,
    ) -> dict[str, float]:
        """计算 ROUGE-N"""
        total_overlap = 0.0
        total_reference = 0.0
        total_candidate = 0.0

        for gen, ref in zip(generated_texts, reference_texts):
            gen_counter = self._get_ngrams(gen, n)
            ref_counter = self._get_ngrams(ref, n)

            overlap = sum((gen_counter & ref_counter).values())
            total_overlap += overlap
            total_reference += sum(ref_counter.values())
            total_candidate += sum(gen_counter.values())

        # 计算 Recall
        recall = total_overlap / total_reference if total_reference > 0 else 0.0

        # 计算 Precision
        precision = total_overlap / total_candidate if total_candidate > 0 else 0.0

        # 计算 F1
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        return {
            f"rouge-{n}_r": recall,
            f"rouge-{n}_p": precision,
            f"rouge-{n}_f": f1,
        }

    def _compute_rouge_l(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 ROUGE-L (最长公共子序列)"""
        total_lcs = 0.0
        total_reference = 0.0
        total_candidate = 0.0

        for gen, ref in zip(generated_texts, reference_texts):
            gen_words = self._tokenize(gen)
            ref_words = self._tokenize(ref)

            lcs = self._lcs_length(gen_words, ref_words)
            total_lcs += lcs
            total_reference += len(ref_words)
            total_candidate += len(gen_words)

        # 计算 Recall
        recall = total_lcs / total_reference if total_reference > 0 else 0.0

        # 计算 Precision
        precision = total_lcs / total_candidate if total_candidate > 0 else 0.0

        # 计算 F1
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        return {
            "rouge-l_r": recall,
            "rouge-l_p": precision,
            "rouge-l_f": f1,
        }

    def compute(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 ROUGE 分数

        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表

        Returns:
            ROUGE 分数字典
        """
        if not generated_texts or not reference_texts:
            return {"rouge-1_r": 0.0, "rouge-1_p": 0.0, "rouge-1_f": 0.0}

        # 根据 variant 选择计算方法
        if self.variant == "rouge-l":
            result = self._compute_rouge_l(generated_texts, reference_texts)
        else:
            # rouge-1, rouge-2, etc.
            n = int(self.variant.split("-")[1])
            result = self._compute_rouge_n(generated_texts, reference_texts, n)

        # 根据 stat 返回相应的值
        stat_key = f"{self.variant}_{self.stat}"
        return result

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        # ROUGE 不适用于检索任务
        return 0.0

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        result = self.compute(generated_texts, reference_texts)
        stat_key = f"{self.variant}_{self.stat}"
        return result.get(stat_key, 0.0)


class RougeMETRIC(BaseModel):
    """ROUGE 指标合集"""

    def compute_rouge_1(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 ROUGE-1"""
        metric = RougeMetric(variant="rouge-1")
        return metric.compute(generated_texts, reference_texts)

    def compute_rouge_2(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 ROUGE-2"""
        metric = RougeMetric(variant="rouge-2")
        return metric.compute(generated_texts, reference_texts)

    def compute_rouge_l(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 ROUGE-L"""
        metric = RougeMetric(variant="rouge-l")
        return metric.compute(generated_texts, reference_texts)

    def compute_all(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算所有 ROUGE 指标"""
        result = {}
        for variant in ["rouge-1", "rouge-2", "rouge-l"]:
            metric = RougeMetric(variant=variant)
            result.update(metric.compute(generated_texts, reference_texts))
        return result


__all__ = ["RougeMetric", "RougeMETRIC"]
