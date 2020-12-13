import shutil
import tarfile
from pathlib import Path

import pandas as pd
import streamlit as st
from directory_tree import display_tree
from smart_open import open


def app():
    st.title("日本語Wikipedia入力誤りデータセット ビューワー")
    st.write("http://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット")

    @st.cache
    def download_data():
        url = "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd.tar.gz&name=JWTD.tar.gz"
        with open(url, "rb") as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall("data")

    text = st.text("Donwloding dataset...")
    download_data()
    text.text(f"Donwload Successed!")

    @st.cache
    def load_dataframe(path):
        return pd.read_json(path, lines=True, orient="records")

    df_train = load_dataframe("data/jwtd/train.jsonl")

    st.write("### File tree")
    st.text(display_tree("data/jwtd", string_rep=True))

    st.sidebar.write("## Display settings")
    cols = st.sidebar.multiselect(
        "Filter columns", list(df_train.columns), default=list(df_train.columns),
    )
    n_data = st.sidebar.selectbox(
        "Max display rows: ", options=[1, 2, 3, 5, 10, 100], index=3
    )
    query = st.sidebar.text_input("query", value="")
    st.sidebar.info(
        "query syntax is based on pandas.query, e.g. category=='kanji-conversion'"
    )

    st.write("### Train.jsonl")
    if query:
        df_view = df_train.query(query)
    else:
        df_view = df_train
    df_view = df_view[cols]

    if st.sidebar.button("Shuffle Table"):
        df_view = df_view.sample(n=n_data)
    st.table(df_view.head(n=n_data))

    st.write("### Stats")
    st.write("Categories")
    st.write(df_train.category.value_counts())
    st.write(df_train.describe())
