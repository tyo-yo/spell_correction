from functools import lru_cache
from itertools import product
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel
from torch.nn.functional import log_softmax
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from spell_correction.utils import BeginningLevenshtein

OVEREDIT = "OVEREDIT"
FINISHED = "FINISHED"
PROCESSING = "PROCESSING"


class Edit(BaseModel):
    text_after: str
    text_before: str
    distance: float
    log_likelihood: float
    likelihood_rank: int


class Beam(BaseModel):
    masked_text: str
    remaining_text: str
    filled_text: str = ""
    total_distance: float = 0.0
    total_log_likelihood: float = 0.0
    changes: List[Edit] = []


class T5SpellCorrector:
    def __init__(
        self,
        model_name="google/mt5-small",
        max_total_distance: int = 2,
        max_each_distance: int = 1,
        substitution_cost_fn: Callable[[str, str], float] = lambda str1, str2: 1.0,
        insertion_cost_fn: Callable[[str], float] = lambda _: 1.0,
        deletion_cost_fn: Callable[[str], float] = lambda _: 1.0,
        score_fn: Callable[[Edit], float] = lambda edit: edit.likelihood,
        filter_fn: Callable[[Edit], bool] = lambda edit: True,
        max_rank: int = 50000,
        add_original_token_to_masked_text: bool = True,
        beam_stack_size: int = 100,
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
        self.score_fn = score_fn
        self.filter_fn = filter_fn
        self.add_original_token_to_masked_text = add_original_token_to_masked_text
        self.beam_stack_size = beam_stack_size

    def predict(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fill_masked_text(self, masked_text: str, original_token: str) -> Dict[str, Any]:
        if self.add_original_token_to_masked_text:
            masked_text += " </s>" + original_token

        beams = self.beam_search(masked_text, original_token)
        return beams[0]
        # return masked_text.replace(" <extra_id_0>", ""), changes

    def beam_search(self, masked_text, original_token) -> List[Beam]:
        beam_stack = [Beam(masked_text=masked_text, remaining_text=original_token)]
        beam_finished = []

        while beam_stack:
            beam = beam_stack.pop()
            possible_beams = self.take_beam_step(beam)
            for possible_beam in possible_beams:
                status = self._get_beam_status(possible_beam)
                if status == FINISHED:
                    beam_finished.append(possible_beam)
                elif status == PROCESSING:
                    beam_stack.append(possible_beam)

            beam_stack = beam_stack[: self.beam_stack_size]
        return beam_finished

    def _get_beam_status(self, beam: Beam) -> str:
        if beam.total_distance > self.max_total_distance:
            return OVEREDIT
        elif len(beam.remaining_str) == 0:
            return FINISHED
        else:
            return PROCESSING

    def _apply_edit_to_beam(self, edit: Edit, beam: Beam) -> Beam:
        beam = beam.copy(deep=True)
        beam.remaining_text = beam.remaining_text[len(edit.text_before) :]
        beam.filled_text += edit.text_after
        beam.total_distance += edit.distance
        beam.total_log_likelihood += edit.log_likelihood
        beam.changes.append(edit)
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
        top_log_likelihoods = log_softmax(top_logits)

        possible_edits = []
        for rank, (token, log_likelihood) in enumerate(
            zip(top_tokens, top_log_likelihoods)
        ):
            if token == self.tokenizer.eos_token:
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

        possible_edits = filter(self.filter_fn, possible_edits)
        possible_edits = sorted(possible_edits, key=self.score_fn, reverse=True)
        return possible_edits

    def take_beam_step(self, beam: Beam) -> List[Beam]:
        possible_edits = self.calc_possible_edits(beam)
        hopeful_edits = self.filter_hopeful_edits(possible_edits)

        possible_beams = [
            self._apply_edit_to_beam(edit, beam) for edit in hopeful_edits
        ]

        return possible_beams

    def filter_hopeful_edits(self, sorted_edits: List[Edit]) -> List[Edit]:
        if not sorted_edits:
            return []
        len_original_token = len(sorted_edits[0].filled_text) + len(
            sorted_edits[0].remaining_text
        )
        hopeful_edits = []
        # 尤度が高く、より一致する部分が多く、編集距離が小さいものがより良いEdit だが、それを選ぶのは難しい
        # そこで、一致度合いと編集距離の各パターンにおいて尤度が最も高くなるEditを全てhopefulとする
        # (match_len, distance) = (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) ... に対して
        for match_len, distance in product(
            range(len_original_token + 1), range(self.max_each_distance + 1)
        ):
            # 条件を満たすものをフィルタした中から、最も尤度が高いものを選択
            # （尤度が高いものからソートされているので、next()で大丈夫）
            hopeful_edit = next(
                filter(
                    lambda edit: len(edit.text_before) == match_len
                    and edit.distance == distance,
                    sorted_edits,
                )
            )
            hopeful_edits.append(hopeful_edit)
        return hopeful_edits
