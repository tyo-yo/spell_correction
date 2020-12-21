# slimだとpillowが動かないため当面こちらを使う
FROM python:3.7.6-slim AS builder

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

RUN apt-get update && apt-get install -y build-essential

# Install MeCab
RUN apt-get update && apt-get install -y mecab libmecab-dev

# cuda install start
# https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/11.0/ubuntu20.04-x86_64/base/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-0=11.0.194-1 \
    cuda-compat-11-0 \
    && ln -s cuda-11.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"

# cuda install end

# Python の依存ライブラリ管理ツールpoetryをインストール
RUN pip install poetry && \
    poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml ${WORKDIR}

# FROM builder as development
RUN poetry install --no-root
# 依存ライブラリのみをインストールし、rootパッケージはインストールしていない
# 明示的に--no-rootをつけているが、rootパッケージ以下をCOPYしてないからこのオプションをつけていなくても結果は同じになる
# pros: rootパッケージ内のコードを書き換えてもbuildし直さなくて良い
# cons: カレントディレクトリが異なる時 import root_package できない（sys.path）

# FROM builder as production
# RUN poetry install --no-dev
# マルチステージビルドを使ってdev-prodの切り分けをするとGA上でcached-buildができなそう？なのでとりあえずやめておく



