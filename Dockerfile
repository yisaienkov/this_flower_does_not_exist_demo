FROM python:3.8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 \
    libopencv-dev libgtk2.0-dev libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

WORKDIR /app/

COPY ./src/ /app/src/
COPY ./resources/ /app/resources/
COPY params.yaml /app/params.yaml
 
CMD streamlit run src/app.py --theme.base dark --server.port $PORT