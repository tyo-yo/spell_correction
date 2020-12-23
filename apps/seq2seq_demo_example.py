from pathlib import Path

import streamlit as st
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader
from chardiff_html import chardiff_html
from spell_correction.predictor import JaSeq2SeqPredictor


def app():
    @st.cache(hash_funcs={JaSeq2SeqPredictor: lambda _: None})
    def load_predictor(path):
        if path.endswith(".tar.gz"):
            return Predictor.from_path(path, "ja_seq2seq")
        elif path.endswith(".th"):
            serialization_dir = str(Path(path).parent)
            params = Params.from_file(str(serialization_dir + "/config.json"))
            model = Model.load(params, str(serialization_dir), weights_file=path)
            dataset_reader = DatasetReader.from_params(params.get("dataset_reader"))
            return JaSeq2SeqPredictor(model, dataset_reader)
        else:
            raise ValueError

    # path = st.sidebar.selectbox(
    #     label="Select Model",
    #     options=[
    #         "https://storage.googleapis.com/tyoyo/experiments/jwtd_test/0000_cpu/model.tar.gz"
    #     ],
    #     format_func=lambda path: "..." + path.replace("/model.tar.gz", "")[-27:],
    # )
    path = st.sidebar.text_input(
        "Model URL or Path",
        value="https://storage.googleapis.com/tyoyo/experiments/jwtd_test/0000_cpu/model.tar.gz",
    )
    predictor = load_predictor(path)

    st.write("### Input")
    input_sentence = st.text_area("Input Sentence", "これが正されるのは、江戸時代に本居宣長の登場してからのことである。")

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
