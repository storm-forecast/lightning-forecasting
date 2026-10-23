# The image ships the system libeccodes, which pygrib needs to read GRIB2.
#
# Python 3.12 is REQUIRED, not incidental: the regressor models carry a Lambda
# layer whose function Keras saved as marshalled CPython bytecode, produced by
# Colab on 3.12. CPython does not keep that format compatible across minor
# versions - on 3.11 the code object comes back truncated (no RETURN_VALUE)
# and calling it segfaults the process; on 3.13 it decodes to garbage.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        libeccodes0 \
        libeccodes-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY worker ./worker
COPY scripts ./scripts
COPY config ./config

# Render's persistent disk is mounted here.
ENV DATA_DIR=/var/data
RUN mkdir -p /var/data

EXPOSE 10000

# A single worker: the pipeline is memory-heavy and keeps in-process state.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --timeout-keep-alive 75"]
