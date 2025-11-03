FROM python:3.9.7-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /code
COPY requirements.txt /code/

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
