import shutil
import tarfile
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
import streamlit as st
from directory_tree import display_tree
from smart_open import open


def app():
    st.title("日本語Wikipedia入力誤りデータセット ビューワー")
    st.write("http://nlp.ist.i.kyoto-u.ac.jp/?日本語Wikipedia入力誤りデータセット")

    def download_data():
        if Path("data/jwtd").exists():
            return
        url = "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd.tar.gz&name=JWTD.tar.gz"
        with open(url, "rb") as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall("data")

    download_data()

    # ここでキャッシュすることで、df_trainなどがリロードしても同一のオブジェクトになり、
    # df_trainを参照している関数のキャッシュがうまく動くようになる
    @st.cache(show_spinner=False)
    def load_dataframe(path):
        return dd.read_json(path, blocksize="1MB")

    df_train = load_dataframe("data/jwtd/train.jsonl")
    df_test = load_dataframe("data/jwtd/test.jsonl")

    st.write("### File tree")
    st.text(display_tree("data/jwtd", string_rep=True))

    st.sidebar.write("## Display settings")
    cols = st.sidebar.multiselect(
        "Filter columns", list(df_train.columns), default=list(df_train.columns),
    )
    n_data = st.sidebar.selectbox(
        "Max display rows: ", options=[1, 2, 3, 5, 10, 100], index=3
    )

    button_placeholder = st.sidebar.empty()
    query = st.sidebar.text_input("query", value="")
    st.sidebar.info(
        "Query syntax is based on pandas.query, e.g. category=='kanji-conversion', post_text.str.contains('キャンパス')"
    )

    st.write("## Browse Train Dataset")

    if query:
        with st.spinner(
            "Query need whole dataset access and slow down browsing. To speed up, remove all queries."
        ):
            df_view = df_train.query(query, engine="python").compute()
    else:
        df_view = df_train
    df_view = df_view[cols]

    if button_placeholder.button("Random Sampling"):
        df_view = df_view.sample(frac=1 / n_data)
    st.table(df_view.head(n_data))

    st.write("## Stats")

    @st.cache
    def calc_stats():
        return dask.compute(
            df_train.size,
            df_test.size,
            df_train.category.value_counts(),
            df_train.describe(),
        )

    n_train, n_test, category_stats, num_stats = calc_stats()

    st.write(f"Train data size: {n_train}, Test data size: {n_test}")
    st.write("Categories")

    st.write(category_stats)
    st.write(num_stats)
