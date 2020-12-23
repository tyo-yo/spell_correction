from typing import List

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric
from Levenshtein import distance
from overrides import overrides


@Metric.register("levenshtein_distance")
class LevenshteinDistance(Metric):
    def __init__(self, avg_by_seq_len: bool = False) -> None:
        self.avg_by_seq_len = avg_by_seq_len
        self._total_value = 0.0
        self._count = 0
        self._total_exact_matches = 0

    @overrides
    def __call__(self, pred_tokens: List[List[str]], gold_tokens: List[List[str]]):
        if is_distributed():
            raise NotImplementedError

        pred_sentences = ["".join(p) for p in pred_tokens]
        gold_sentences = ["".join(g) for g in gold_tokens]

        distances = [distance(p, g) for p, g in zip(pred_sentences, gold_sentences)]

        self._total_exact_matches += sum([d == 0 for d in distances])

        self._count += len(distances)

        if self.avg_by_seq_len:
            self._total_value += sum(
                [d / len(g) for d, g in zip(distances, gold_sentences)]
            )
        else:
            self._total_value += sum(distances)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns
        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0.0
        avg_exact_match = self._total_exact_matches / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return {
            "LevenshteinDistance": float(average_value),
            "ExactMatch": avg_exact_match
        }

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
        self._total_exact_matches = 0
