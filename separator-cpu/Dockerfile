FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchaudio==2.4.1
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch
CMD ["python", "-u", "/app/handler.py"]
