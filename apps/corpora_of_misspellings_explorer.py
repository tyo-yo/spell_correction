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
    st.title("Corpora of misspellings Explorer")
    st.write("https://www.dcs.bbk.ac.uk/~ROGER/corpora.html")
    st.write("")

    @st.cache
    def download_data():
        # if Path("data/jwtd").exists():
        #     return
        url = "https://www.dcs.bbk.ac.uk/~ROGER/missp.dat"
        with open(url, "r") as f:
            df = pd.read_csv(f, header=None)
            # with tarfile.open(fileobj=f) as tar:
            #     tar.extractall("data")
        return df

    with st.spinner("Donwloading Dataset..."):
        df = download_data()

    st.write("birkbeck.dat, 36,133 misspellings of 6,136 word")
    st.write(df.head(n=100))

    st.write("$America の後に Americaのタイポが並ぶようなデータ形式になっている")
    st.write(
        "[この研究](https://www.aclweb.org/anthology/2020.findings-emnlp.37.pdf)とかでこのデータが使われている"
    )
