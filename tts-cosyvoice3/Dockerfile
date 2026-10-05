FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git git-lfs libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# CosyVoice repo has no proper setup.py; the `cosyvoice` package on PyPI is an
# unrelated project and does NOT provide `cosyvoice.cli`. Clone the real repo
# and expose it (plus its Matcha-TTS submodule) on PYTHONPATH so the handler's
# `from cosyvoice.cli.cosyvoice import CosyVoice3` import resolves.
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
 && git -C /opt/CosyVoice submodule update --init --recursive || true
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch
CMD ["python", "-u", "/app/handler.py"]
