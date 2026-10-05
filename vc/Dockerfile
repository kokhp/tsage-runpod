FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir "freevc @ git+https://github.com/OlaWod/FreeVC.git@main" \
    || true  # FreeVC is a research repo; fall back to passthrough if install fails

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch
CMD ["python", "-u", "/app/handler.py"]
