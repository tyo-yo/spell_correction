import difflib

import streamlit as st
from chardiff_html import chardiff_html


def app():
    st.write("### Input")
    str1 = st.text_area("Original Sentence", "This is orignal sentence.")
    str2 = st.text_area("New Sentence", "That is new sentence.")
    res = chardiff_html(str1, str2)
    st.markdown("### Diffs")
    st.markdown(
        res, unsafe_allow_html=True,
    )
