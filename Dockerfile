FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /usr/app

ENV PIP_DEFAULT_TIMEOUT=500

USER root
ARG DEBIAN_FRONTEND=noninteractive

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
    apt-get update && \
    apt install ffmpeg libsndfile1 build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
RUN apt-get install -y git
RUN apt install python3-pip -y
RUN apt-get install git-lfs
RUN git clone https://huggingface.co/wmpscc/StyleGene stylegene/checkpoints
RUN git clone https://huggingface.co/datasets/nuwandaa/ffhq128 stylegene/data/thumbnails128x128
RUN apt-get install unzip
RUN unzip stylegene/data/thumbnails128x128/thumbnails128x128.zip -d stylegene/data/thumbnails128x128
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib"

COPY requirements.txt /usr/app/requirements.txt
RUN pip install -r requirements.txt
ENV USE_GENE_POOL="false"
ENV MAX_SAMPLES="50"
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "3","--log-level", "info"]