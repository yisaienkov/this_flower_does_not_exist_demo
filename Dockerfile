FROM python:3.8

COPY ./requirements.txt /requirements.txt

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

WORKDIR /app/

COPY ./src/ /app/src/
COPY ./resources/ /app/resources/
COPY params.yaml /app/params.yaml
 
CMD streamlit run src/app.py --theme.base dark --server.port $PORT