FROM python:3.8

COPY ./src/ app/src/
COPY ./requirements.txt app/requirements.txt

WORKDIR /app/

RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt && \
    python -m pip cache purge

 
CMD streamlit run src/app.py --theme.base dark --server.port $PORT