FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install streamlit requests pandas opencv-python Pillow

WORKDIR /landmarks_app
COPY . /landmarks_app