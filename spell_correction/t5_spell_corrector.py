import math
from functools import lru_cache
from itertools import product
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from scipy.stats import hmean
from torch.nn.functional import log_softmax
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from spell_correction.utils import BeginningLevenshtein

logger = getLogger(__name__)
OVEREDIT = "OVEREDIT"
FINISHED = "FINISHED"
PROCESSING = "PROCESSING"


class Edit(BaseModel):
    text_after: str
    text_before: str
    distance: float
    log_likelihood: float = 0  # TODO: 本当は違う！！
    likelihood_rank: int = -1


class Beam(BaseModel):
    masked_text: str
    remaining_text: str
    original_text: str
    filled_text: str = ""
    total_distance: float = 0.0
    total_log_likelihood: float = 0.0
    changes: List[Edit] = []
    score: float = 0.0


def harmonic_mean(beam: Beam, max_distance: float) -> float:
    # 0~1のスコアで、高いほど良い。
    # max_distance * 2 にすることで、0.5 ~ 1.0 くらいに収める
    # distance = max_distance で0になってしまうと、hmeanも0になってしまうからあまり嬉しくない
    distance_score = 1 - beam.total_distance / (max_distance * 2)

    # (len(beam.original_text) * 2)にすることで、 0.5 ~1.0 を値域とするスコアとする
    noramlized_match_len = 1 - len(beam.remaining_text) / (len(beam.original_text) * 2)
    # likelihood = math.exp(beam.total_log_likelihood)
    # だいたい-1~-100くらいだからという雑な正規化
    likelihood = 1 - abs(beam.total_log_likelihood) / 100
    vals = [distance_score, noramlized_match_len, likelihood]
    vals = [np.clip(v, 0, 1) for v in vals]

    return hmean(vals)
    return 3 / (1 / distance_score + 1 / noramlized_match_len + 1 / likelihood)


class T5SpellCorrector:
    def __init__(
        self,
        model_name="google/mt5-small",
        max_total_distance: int = 2,
        max_each_distance: int = 1,
        substitution_cost_fn: Callable[[str, str], float] = lambda str1, str2: 1.0,
        insertion_cost_fn: Callable[[str], float] = lambda _: 1.0,
        deletion_cost_fn: Callable[[str], float] = lambda _: 1.0,
        max_rank: int = 50000,
        add_original_token_to_masked_text: bool = False,
        beam_size: int = 5,
    ):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_each_distance = max_each_distance
        self.max_total_distance = max_total_distance

        self.beginning_levenshtein = BeginningLevenshtein(
            substitution_cost_fn=substitution_cost_fn,
            insertion_cost_fn=insertion_cost_fn,
            deletion_cost_fn=deletion_cost_fn,
        )
        self.beginning_levenshtein.__call__ = lru_cache(
            maxsize=self.tokenizer.vocab_size ** 2
        )(self.beginning_levenshtein.__call__)

        self.max_rank = max_rank
        self.add_original_token_to_masked_text = add_original_token_to_masked_text
        self.beam_size = beam_size

    def predict(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fill_masked_text(self, masked_text: str, original_token: str) -> Dict[str, Any]:
        beams = self.beam_search(masked_text, original_token)
        return beams[0]
        # return masked_text.replace(" <extra_id_0>", ""), changes

    def beam_search(self, masked_text, original_token, beam_size=None) -> List[Beam]:
        if beam_size is None:
            beam_size = self.beam_size
        if self.add_original_token_to_masked_text:
            masked_text += " </s>" + original_token
        beam_stack = [
            Beam(
                masked_text=masked_text,
                remaining_text=original_token,
                original_text=original_token,
            )
        ]
        next_beam_stack = []
        beam_finished = []

        while beam_stack:
            beam = beam_stack.pop()
            possible_beams = self.take_beam_step(beam)
            for possible_beam in possible_beams:
                status = self._get_beam_status(possible_beam)
                if status == FINISHED:
                    logger.debug(possible_beam)
                    beam_finished.append(possible_beam)
                elif status == PROCESSING:
                    next_beam_stack.append(possible_beam)

            if not beam_stack and next_beam_stack:
                # パレート最適なビームに絞る
                beam_stack = self.filter_hopeful_beams(next_beam_stack)
                # パレート最適な候補がbeam_sizeより多い場合、調和平均の高いものから選ぶ
                beam_stack.sort(key=lambda b: b.score, reverse=True)
                beam_stack = beam_stack[:beam_size]
                next_beam_stack = []

        return beam_finished

    def _get_beam_status(self, beam: Beam) -> str:
        if beam.total_distance > self.max_total_distance:
            return OVEREDIT
        # TODO: 全部マッチしたら終了だと末尾にinsertできない
        elif len(beam.remaining_text) == 0:
            return FINISHED
        else:
            return PROCESSING

    def _apply_edit_to_beam(self, edit: Edit, beam: Beam) -> Beam:
        beam = beam.copy(deep=True)
        beam.remaining_text = beam.remaining_text[len(edit.text_before) :]
        beam.filled_text += edit.text_after
        if beam.remaining_text:
            beam.total_distance = self.beginning_levenshtein.distance(
                beam.filled_text, beam.original_text
            )
        else:
            beam.total_distance = self.beginning_levenshtein.weighted_levenshtein.distance(
                beam.filled_text, beam.original_text
            )
        # TODO: status==FINISHED の場合、次の単語がEOSであるという尤度の計算が抜けているので追加する。
        beam.total_log_likelihood += edit.log_likelihood
        beam.changes.append(edit)
        beam.score = harmonic_mean(beam, self.max_total_distance)
        return beam

    def calc_possible_edits(self, beam: Beam) -> List[Edit]:
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=[beam.masked_text],
            tgt_texts=["<pad> <extra_id_0>" + beam.filled_text],
            return_tensors="pt",
        )

        output = self.model(**batch)
        batch_top_logits, batch_top_indices = output.logits[:, -1].sort(
            dim=-1, descending=True
        )  # 末尾のトークンの次にくる単語の予測をスコアでソート
        batch_top_tokens = [
            self.tokenizer.convert_ids_to_tokens(each_indices)
            for each_indices in batch_top_indices.tolist()
        ]

        # batch size = 1 前提のコード
        top_tokens = batch_top_tokens[0]
        top_logits = batch_top_logits[0]
        top_log_likelihoods = log_softmax(top_logits, dim=0)

        possible_edits = []
        for rank, (token, log_likelihood) in enumerate(
            zip(top_tokens, top_log_likelihoods)
        ):
            if token == self.tokenizer.eos_token or token.startswith("▁<extra_id_"):
                possible_edit = Edit(
                    text_after="",
                    text_before=beam.remaining_text,
                    distance=len(beam.remaining_text),
                    log_likelihood=log_likelihood.tolist(),
                    likelihood_rank=rank,
                )
            else:
                diff = self.beginning_levenshtein(token, beam.remaining_text)
                possible_edit = Edit(
                    text_after=token,
                    text_before=diff["match_ref"],
                    distance=diff["distance"],
                    log_likelihood=log_likelihood.tolist(),
                    likelihood_rank=rank,
                )
            possible_edits.append(possible_edit)

            # 計算量を減らすために尤度が低いものは足切り
            if rank >= self.max_rank:
                break
        return possible_edits

    def take_beam_step(self, beam: Beam) -> List[Beam]:
        possible_edits = self.calc_possible_edits(beam)
        hopeful_edits = self.filter_hopeful_edits(possible_edits, beam)

        possible_beams = [
            self._apply_edit_to_beam(edit, beam) for edit in hopeful_edits
        ]

        return possible_beams

    def filter_hopeful_edits(self, sorted_edits: List[Edit], beam: Beam) -> List[Edit]:
        if not sorted_edits:
            return []
        hopeful_edits = []
        max_distance = min(
            self.max_each_distance, self.max_total_distance - beam.total_distance,
        )
        available_distances = set(
            [edit.distance for edit in sorted_edits if edit.distance <= max_distance]
        )
        available_match_len = range(len(beam.original_text) + 1)
        # 尤度が高く、より一致する部分が多く、編集距離が小さいものがより良いEdit だが、それを選ぶのは難しい
        # そこで、一致度合いと編集距離の各パターンにおいて尤度が最も高くなるEditを全てhopefulとする
        # (match_len, distance) = (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) ... に対して
        for match_len, distance in product(available_match_len, available_distances):
            # 条件を満たすものをフィルタした中から、最も尤度が高いものを選択
            # （尤度が高いものからソートされているので、next()で大丈夫）
            filtered_edits = list(
                filter(
                    lambda edit: len(edit.text_before) == match_len
                    and edit.distance == distance,
                    sorted_edits,
                )
            )
            if not filtered_edits:
                continue
            hopeful_edit = max(filtered_edits, key=lambda edit: edit.log_likelihood)
            hopeful_edits.append(hopeful_edit)
        return hopeful_edits

    def filter_hopeful_beams(self, beams: List[Beam]) -> List[Beam]:
        # 編集距離は短いほどよい
        unique_distances = sorted(set([b.total_distance for b in beams]))
        # まだマッチしていない長さは短いほどいい (=マッチしている長さは長いほど良い)
        unique_remaining_len = sorted(set([len(b.remaining_text) for b in beams]))

        # パレート最適なビームを取得
        pareto_optimal_beams = []
        for distance, remaining_len in product(unique_distances, unique_remaining_len):
            filtered_beams = list(
                filter(
                    lambda beam: len(beam.remaining_text) == remaining_len
                    and beam.total_distance == distance,
                    beams,
                )
            )
            if filtered_beams:
                pareto_optimal_beams.append(
                    max(filtered_beams, key=lambda beam: beam.total_log_likelihood)
                )
        return pareto_optimal_beams
