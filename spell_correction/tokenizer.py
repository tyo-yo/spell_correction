from typing import List, Optional

import MeCab
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from overrides import overrides


@Tokenizer.register("mecab")
class MeCabTokenizer(Tokenizer):
    def __init__(
        self,
        dicdir: str = "",
        userdic: str = "",
        other_options: str = "",
        special_tokens: Optional[List[str]] = None,
    ) -> None:

        self._special_tokens = special_tokens or ()

        options = "-Owakati"

        if dicdir:
            options += f" -d {dicdir}"
        if userdic:
            options += f" -u {userdic}"
        if other_options:
            options += f" {other_options}"

        self.mecab = MeCab.Tagger(options)

    @overrides
    def tokenize(self, sentence: str) -> List[Token]:
        tokens: List[Token] = []
        # split by special tokens
        if self._special_tokens:
            segments = split_by_words(sentence, self._special_tokens)
        else:
            segments = [sentence]

        # tokenize
        elems = []
        for segment in segments:
            if not segment:
                continue
            elif segment in self._special_tokens:
                elems.append(segment)
            else:
                elems.extend(self.mecab.parse(segment).strip().split())

        tokens = [Token(text=elem) for elem in elems]
        return tokens


def split_by_words(sentence: str, words: List[str]) -> List[str]:
    """
    Splits sentence by words.
    Parameters
    ----------
    sentence : `str`
        Input string.
    words : `List[str]`
        Words to split the input sentence.
    keep_deliminator : `bool`, optional (default = True)
        Determines whether to keep deliminator (`words`).
    Returns
    segments : `List[str]`
        Split sentence.
    """
    segments = [sentence]
    for word in words:
        _tmp_segments = []
        for segment in segments:
            _new_seg = []
            for _seg in segment.split(word):
                _new_seg.append(word)
                if _seg:
                    _new_seg.append(_seg)
            _new_seg = _new_seg[1:]
            _tmp_segments.extend(_new_seg)
        segments = _tmp_segments
    return segments
