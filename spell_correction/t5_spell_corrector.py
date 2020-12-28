from functools import lru_cache
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from spell_correction.utils import BeginningLevenshtein, sigmoid


def default_score_func(candidate):
    return (
        candidate["match_len"]
        / (candidate["distance"] + 1)
        * sigmoid(candidate["logit"])
    )


def default_filter_func(candidate):
    return candidate["distance"] <= 2 and candidate["match_len"] >= 1


class Edit(BaseModel):
    str_after: str
    str_before: str
    distance: float
    likelihood: float
    likelihood_rank: int


class Beam(BaseModel):
    masked_text: str
    remaining_str: str
    total_distance: float = 0.0
    total_logits: float = 0.0
    changes: List[Edit] = []


class T5SpellCorrector:
    def __init__(
        self,
        model_name="google/mt5-small",
        max_edit_distance: int = 2,
        substitution_cost_fn=lambda str1, str2: 1,
        insertion_cost_fn=lambda _: 1,
        deletion_cost_fn=lambda _: 1,
        max_decoding_steps: int = 10,
        max_rank: int = 50000,
        score_func=default_score_func,
        filter_func=default_filter_func,
        add_original_token_to_masked_text: bool = True,
    ):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_edit_distance = max_edit_distance

        self.beginning_levenshtein = BeginningLevenshtein(
            substitution_cost_fn=substitution_cost_fn,
            insertion_cost_fn=insertion_cost_fn,
            deletion_cost_fn=deletion_cost_fn,
        )
        self.beginning_levenshtein.__call__ = lru_cache(
            maxsize=self.tokenizer.vocab_size ** 2
        )(self.beginning_levenshtein.__call__)

        self.max_decoding_steps = max_decoding_steps
        self.max_rank = max_rank
        self.score_func = score_func
        self.filter_func = filter_func
        self.add_original_token_to_masked_text = add_original_token_to_masked_text

    def predict(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fill_masked_text(self, masked_text: str, original_token: str) -> Dict[str, Any]:
        if self.add_original_token_to_masked_text:
            masked_text += " </s>" + original_token

        beams = self.beam_search(masked_text, original_token)
        return beams[0]
        # return masked_text.replace(" <extra_id_0>", ""), changes

    def beam_search(self, masked_text, original_token) -> List[Dict[str, Any]]:
        beam_stack = [Beam(masked_text=masked_text, remaining_str=original_token)]
        beam_finished = []

        while beam_stack:
            beam = beam_stack.pop()
            possible_beams = self._process_beam(beam)
            for possible_beam in possible_beams:
                if self._is_finished_beam(possible_beam):
                    beam_finished.append(possible_beam)
                else:
                    beam_stack.append(possible_beam)

    def _process_beam(self, beam: Beam) -> List[Beam]:
        raise NotImplementedError
        remaining_str = original_token
        changes = []

        for step in range(self.max_decoding_steps):
            candidates = self.calc_next_token_candidates(masked_text, remaining_str)
            selected = candidates[0]
            changes.append(selected)

            masked_text = masked_text.replace(
                " <extra_id_0>", f"{selected['token']} <extra_id_0>"
            )
            remaining_str = remaining_str[selected["match_len"] :]
            if remaining_str == "":
                break

    def _is_finished_beam(self, beam: Beam) -> List[Beam]:
        return len(beam.remaining_str) == 0

    def calc_next_token_candidates(self, masked_text, remaining_str):
        batch = self.tokenizer.prepare_seq2seq_batch(
            src_texts=[masked_text],
            tgt_texts=["<pad> <extra_id_0>"],
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
        top_indices = batch_top_indices[0]

        candidates = []
        for rank, (token, logit, index) in enumerate(
            zip(top_tokens, top_logits, top_indices)
        ):
            if token == self.tokenizer.eos_token:
                candidate = {
                    "match_ref": remaining_str,
                    "match_len": len(remaining_str),
                    "distance": len(remaining_str),
                    "rank": rank,
                    "token": "",
                    "logit": logit.tolist(),
                    "index": index.tolist(),
                }
            else:
                candidate = self.beginning_levenshtein(token, remaining_str)
                candidate["rank"] = rank
                candidate["token"] = token
                candidate["logit"] = logit.tolist()
                candidate["index"] = index.tolist()
            candidates.append(candidate)
            if rank >= self.max_rank:
                break

        candidates = filter(self.filter_func, candidates)
        candidates = sorted(candidates, key=self.score_func, reverse=True)
        return candidates
