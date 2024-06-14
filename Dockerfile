# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with source build for CPU

FROM mcr.microsoft.com/cbl-mariner/base/python:3
MAINTAINER Changming Sun "chasun@microsoft.com"
ADD . /code

RUN tdnf install -y tar ca-certificates build-essential cmake curl python3-devel python3-setuptools python3-wheel python3-pip python3-numpy python3-flatbuffers python3-packaging python3-protobuf
# The latest cmake version in Mariner2 is 3.21, but we need 3.26+
RUN /code/dockerfiles/scripts/install_cmake.sh

# Prepare onnxruntime repository & build onnxruntime
RUN cd /code && /bin/bash ./build.sh --allow_running_as_root --skip_submodule_sync --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER)

FROM mcr.microsoft.com/cbl-mariner/base/python:3
COPY --from=0 /code/build/Linux/Release/dist /root
COPY --from=0 /code/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt
RUN tdnf install -y ca-certificates python3-setuptools python3-wheel python3-pip python3-numpy python3-flatbuffers python3-packaging python3-protobuf python3-mpmath python3-sympy && python3 -m pip install coloredlogs humanfriendly && python3 -m pip install --no-index --find-links /root onnxruntime  && rm -rf /root/*.whl

ARG FACEFUSION_VERSION=2.6.0
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/app

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install git-lfs
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y

RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch .
RUN python install.py --onnxruntime default --skip-conda

COPY requirements.txt /usr/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow
RUN pip install typing-extensions==4.9.0 --upgrade
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "3","--log-level", "info"]