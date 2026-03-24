FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install "numpy<2" \
    && python -m pip install -r requirements.txt

COPY app.py ./app.py
COPY src ./src

RUN mkdir -p /app/data /app/faiss_store /app/faiss_store_images /cache/huggingface

ENTRYPOINT ["python", "app.py"]
