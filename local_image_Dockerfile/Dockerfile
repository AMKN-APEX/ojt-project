FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /code

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-pip \
       python3-venv \
       build-essential \
       libgl1 \
       ca-certificates \
       wget \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 || true

