FROM runpod/pytorch:2.4.1-cuda12.1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir "parler-tts @ git+https://github.com/huggingface/parler-tts.git@main" \
    || pip install --no-cache-dir parler-tts

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch

CMD ["python", "-u", "/app/handler.py"]
