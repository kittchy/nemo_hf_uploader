FROM python:3.10.13-slim
ENV TZ=Asia/Tokyo 
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
  apt-get upgrade -y && \
  apt-get install -y \
  build-essential \
  zlib1g-dev \
  libncurses5-dev \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  libsqlite3-dev \
  libbz2-dev \
  software-properties-common \
  tzdata \
  "python3.10" \
  python3-pip \
  wget \
  git \
  curl \
  libsndfile1 \
  libsndfile1-dev \
  ffmpeg \
  git-lfs \
  vim && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip wheel setuptools && \
  pip3 install \
  torch \
  torchvision \
  torchaudio \
  Cython \
  "nemo_toolkit[all]" \
  text-unidecode \
  huggingface-hub \
  classopt \
  loguru \
  omegaconf \
  hydra-core \
  cython \
  git+https://github.com/LahiLuk/YouTokenToMe.git
RUN git lfs install

ENV PYTHONPATH=/app
WORKDIR /app
COPY ../src/ /app/src/
