FROM python:3.10

ARG FACEFUSION_VERSION=2.6.0
ENV GRADIO_SERVER_NAME=0.0.0.0

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