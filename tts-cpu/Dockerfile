FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git libsndfile1 build-essential espeak-ng \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# CPU torch wheels keep the image small enough to boot in a 4-vCPU pod
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchaudio==2.4.1
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir \
    "melotts @ git+https://github.com/myshell-ai/MeloTTS.git@main" \
    && python -m unidic download || true

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf

CMD ["python", "-u", "/app/handler.py"]
