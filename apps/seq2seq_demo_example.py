import difflib
import importlib

import spacy

# if importlib.util.find_spec("spell_correction") is None:
import spell_correction
import streamlit as st
from allennlp.predictors import Predictor
from chardiff_html import chardiff_html
from spell_correction.predictor import JaSeq2SeqPredictor


def app():
    @st.cache(hash_funcs={JaSeq2SeqPredictor: lambda _: None})
    def load_model(archive_path):
        return Predictor.from_path(archive_path, "ja_seq2seq")

    st.write("### Input")
    input_sentence = st.text_area("Input Sentence", "これが正されるのは、江戸時代に本居宣長の登場してからのことである。")

    predictor = load_model("experiments/jwtd_test/0000_cpu/model.tar.gz")
    output_json = predictor.predict(input_sentence)
    output_sentence = "".join(output_json["predicted_tokens"])

    st.write("### Output")
    st.write(output_sentence)

    st.markdown("### Diffs")
    res = chardiff_html(input_sentence, output_sentence)
    st.markdown(
        res, unsafe_allow_html=True,
    )

    st.markdown("### Detailed Ouput Json")
    st.write(output_json)
