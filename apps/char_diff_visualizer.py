import difflib

import streamlit as st


def app():
    st.write("## Input")
    str1 = st.text_area("Original Sentence", "This is orignal sentence.")
    str2 = st.text_area("New Sentence", "That is new sentence.")
    res = visualize_chardiff(str1, str2)
    st.markdown("## Diffs")
    st.markdown(
        res, unsafe_allow_html=True,
    )


def visualize_chardiff(
    str1,
    str2,
    del_font_color="red",
    del_background_color="mistyrose",
    ins_font_color="green",
    ins_background_color="#e0ffe5",
    **kwargs,
):
    # State meaning:: ' ': equal, '+': insert, '-': delete
    prev_state, state = " ", " "
    output = ""
    for diff in difflib.ndiff(str1, str2, **kwargs):
        prev_state = state
        state, _, char = diff

        if state == prev_state:
            output += char
            continue

        if prev_state != " ":
            output += "</span>"

        if state == "+":
            output += f'<span style="color: {ins_font_color}; background-color: {ins_background_color}">'
        elif state == "-":
            output += f'<span style="color: {del_font_color}; background-color: {del_background_color}">'
        output += char

    if state != " ":
        output += "</span>"
    return output
