"""BLEU 评估指标

对齐 WeKnora99 BLEU 实现。
"""

import math
from collections import Counter
from typing import Any

from pydantic import BaseModel


class BLEUMetric(BaseModel):
    """BLEU (Bilingual Evaluation Understudy) 指标

    用于评估生成文本与参考文本的相似度。
    广泛应用于机器翻译和文本生成任务。
    """

    max_n: int = 4  # 最大 n-gram 阶数
    smoothing: bool = True

    @property
    def name(self) -> str:
        return "bleu"

    def _get_ngrams(self, text: str, n: int) -> Counter:
        """获取 n-gram 计数"""
        words = text.lower().split()
        if len(words) < n:
            return Counter()
        return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    def _brevity_penalty(self, candidate_length: int, reference_length: int) -> float:
        """计算简洁性惩罚"""
        if candidate_length == 0:
            return 0.0
        if candidate_length >= reference_length:
            return 1.0
        return math.exp(1 - reference_length / candidate_length)

    def compute(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> dict[str, float]:
        """计算 BLEU 分数

        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表

        Returns:
            BLEU 分数字典
        """
        if not generated_texts or not reference_texts:
            return {"bleu-1": 0.0, "bleu-2": 0.0, "bleu-4": 0.0}

        # 累加统计
        total_precision = {n: 0.0 for n in range(1, self.max_n + 1)}
        total_count = {n: 0.0 for n in range(1, self.max_n + 1)}
        total_reference_length = 0
        total_candidate_length = 0

        for gen, ref in zip(generated_texts, reference_texts):
            gen_words = gen.lower().split()
            ref_words = ref.lower().split()

            candidate_length = len(gen_words)
            reference_length = len(ref_words)

            total_candidate_length += candidate_length
            total_reference_length += reference_length

            # 计算各阶 n-gram 精确率
            gen_ngrams = [self._get_ngrams(gen, n) for n in range(1, self.max_n + 1)]
            ref_ngrams = [self._get_ngrams(ref, n) for n in range(1, self.max_n + 1)]

            for n in range(1, self.max_n + 1):
                gen_counter = gen_ngrams[n - 1]
                ref_counter = ref_ngrams[n - 1]

                # 计算匹配数
                matches = sum((gen_counter & ref_counter).values())

                # 计算总候选 n-gram 数
                count = sum(gen_counter.values())

                total_precision[n] += matches
                total_count[n] += count

        # 计算各阶精确率
        precisions = {}
        for n in range(1, self.max_n + 1):
            if total_count[n] > 0:
                precisions[f"p{n}"] = total_precision[n] / total_count[n]
            else:
                precisions[f"p{n}"] = 0.0

        # 平滑处理
        if self.smoothing:
            for n in range(1, self.max_n + 1):
                if precisions[f"p{n}"] == 0:
                    precisions[f"p{n}"] = 1e-10  # 平滑

        # 计算几何平均
        log_precisions = [math.log(precisions[f"p{n}"]) for n in range(1, self.max_n + 1)]
        avg_precision = math.exp(sum(log_precisions) / self.max_n)

        # 计算简洁性惩罚
        bp = self._brevity_penalty(total_candidate_length, total_reference_length)

        # 计算 BLEU
        bleu = bp * avg_precision

        # 计算各阶 BLEU
        bleu_scores = {"bleu": bleu}
        for n in range(1, self.max_n + 1):
            key = f"bleu-{n}"
            p = precisions[f"p{n}"]
            if p > 0:
                score = bp * p
            else:
                score = 0.0
            bleu_scores[key] = round(score, 4)

        return bleu_scores

    def compute_retrieval(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
    ) -> float:
        # BLEU 不适用于检索任务
        return 0.0

    def compute_generation(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
    ) -> float:
        result = self.compute(generated_texts, reference_texts)
        return result.get("bleu", 0.0)


__all__ = ["BLEUMetric"]
