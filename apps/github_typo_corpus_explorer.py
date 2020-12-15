import json
import re
import shutil
import tarfile
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
import streamlit as st
from chardiff_html import chardiff_html
from directory_tree import display_tree
from smart_open import open


def app():
    st.title("Github Typo Corpus Explorer")
    st.write("https://github.com/mhagiwara/github-typo-corpus")
    st.write("more than 350k edits, 65M characters in more than 15 language")

    @st.cache
    def download_dataframe(n_rows=10000):
        # if Path("data/jwtd").exists():
        #     return
        url = "https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz"
        dataset = []
        for i, line in enumerate(open(url)):
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(dataset) >= n_rows:
                break
            # df = pd.read_json(f, orient="records", lines=True)
            # df = dd.read_json(f).sample(frac=0.01).compute()
            # raw_data = json.load(f)
        meta_cols = list(dataset[0].keys())
        meta_cols.remove("edits")
        df = pd.json_normalize(dataset, record_path="edits", meta=meta_cols, sep="_")
        return df

    st.sidebar.write("## Display settings")

    n_rows = st.sidebar.selectbox("Donwload rows: ", options=[1000, 10000], index=0)

    n_display = st.sidebar.selectbox(
        "Display rows: ", options=[1, 2, 5, 10, 100], index=3
    )

    df = download_dataframe(n_rows)

    cols = st.sidebar.multiselect(
        "Filter columns", list(df.columns), default=list(df.columns),
    )

    st.sidebar.button("Random Sampling")
    query = st.sidebar.text_input("Query (approximate)", value="")
    with st.sidebar.beta_expander("Query exmaples"):
        st.markdown(
            """
        * src_lang != 'eng'
        * 5 < src_ppl < 10
        * not src_text.str.contains('\\\.')
        * src_path.str.contains('.txt')
        """
        )
    st.sidebar.info("Query syntax is based on pandas.query")

    if query:
        df = df.query(query, engine="python")

    df = df.sample(n=n_display)
    st.table(df[cols])

    st.write("## Diffs \n ---")
    for _, row in df.iterrows():
        diff = chardiff_html(row.src_text, row.tgt_text)
        st.markdown(diff, unsafe_allow_html=True)
