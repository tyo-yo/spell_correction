import json
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


def japanese_predictor(cls):
    """日本語対応したload_line, dump_line関数に置き換えるためのデコレータ"""

    def load_line(self, line: str) -> JsonDict:
        return json.loads(line, encoding="utf-8")

    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs, ensure_ascii=False) + "\n"

    setattr(cls, "load_line", load_line)
    setattr(cls, "dump_line", dump_line)


@japanese_predictor
@Predictor.register("ja_seq2seq")
class JaSeq2SeqPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

