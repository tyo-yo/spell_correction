import importlib
from pathlib import Path

import streamlit as st

# apps以下のファイルを取得
app_names = [
    path.with_suffix("").name
    for path in Path("apps").iterdir()
    if not path.name.startswith("_")
]

PAGES = {}

# 各アプリのapp()関数を呼び出せるようにmoduleへの参照を保存しておく
for app_name in app_names:
    module = importlib.import_module(f"apps.{app_name}")
    PAGES[app_name] = module


st.sidebar.title("Pages")
selection = st.sidebar.radio("Go to", app_names)
page = PAGES[selection]
# apps以下の変更が逐次反映されるようにreloadする
importlib.reload(page)
page.app()

