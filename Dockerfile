FROM nvidia/cuda:11.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /tmp/

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libbz2-dev libdb-dev \
    libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
    libncursesw5-dev libsqlite3-dev libssl-dev \
    zlib1g-dev uuid-dev tk-dev \
    # MeCab
    mecab \
    libmecab-dev

# Install Python
RUN wget https://www.python.org/ftp/python/3.7.10/Python-3.7.10.tgz && \
    tar xvf Python-3.7.10.tgz

WORKDIR /tmp/Python-3.7.10
RUN ./configure && \
    make && \
    make install

WORKDIR /usr/local/bin
RUN ln -s python3.7 python

ENV WORKDIR /poc_qa/
WORKDIR ${WORKDIR}

# Python の依存ライブラリ管理ツールpoetryをインストール
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH $PATH:/root/.poetry/bin
RUN poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ${WORKDIR}

# FROM builder as development
RUN poetry install --no-root
# 依存ライブラリのみをインストールし、rootパッケージはインストールしていない
# 明示的に--no-rootをつけているが、rootパッケージ以下をCOPYしてないからこのオプションをつけていなくても結果は同じになる
# pros: rootパッケージ内のコードを書き換えてもbuildし直さなくて良い
# cons: カレントディレクトリが異なる時 import root_package できない（sys.path）

# jupyter上でhboxを表示するため
# https://stackoverflow.com/questions/57343134/jupyter-notebooks-not-displaying-progress-bars
# https://qiita.com/seibe/items/36cef7df85fe2cefa3ea
RUN apt-get update & apt-get install -y nodejs ;\
    apt-get install -y npm ;\
    npm install n -g ;\
    n stable ;\
    apt purge -y nodejs npm ;\
    jupyter nbextension enable --py widgetsnbextension ;\
    jupyter labextension install @jupyter-widgets/jupyterlab-manager


