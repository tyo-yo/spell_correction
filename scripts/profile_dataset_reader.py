from allennlp.data.tokenizers import SpacyTokenizer
from allennlp_models.generation.dataset_readers.seq2seq import Seq2SeqDatasetReader
from spell_correction.tokenizer import MeCabTokenizer


def test_profile():
    data_path = "https://storage.googleapis.com/tyoyo/jwtd/v1.0/dev.tsv"
    dataset_reader = Seq2SeqDatasetReader(
        source_tokenizer=SpacyTokenizer(language="ja_core_news_sm"),
        target_tokenizer=SpacyTokenizer(language="ja_core_news_sm"),
        source_max_tokens=64,
        target_max_tokens=64,
        start_symbol="STARTSYMBOL",
        end_symbol="ENDSYMBOL",
    )
    dataset = dataset_reader.read(data_path)


def test_profile_mecab():
    data_path = "https://storage.googleapis.com/tyoyo/jwtd/v1.0/dev.tsv"
    dataset_reader = Seq2SeqDatasetReader(
        source_tokenizer=MeCabTokenizer(),
        target_tokenizer=MeCabTokenizer(),
        source_max_tokens=64,
        target_max_tokens=64,
        start_symbol="STARTSYMBOL",
        end_symbol="ENDSYMBOL",
    )
    dataset = dataset_reader.read(data_path)
