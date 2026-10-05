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
# CosyVoice's own runtime deps (hyperpyyaml, conformer, wetext, modelscope, …)
# live in its requirements.txt. Install them so `cosyvoice.cli.cosyvoice`
# imports cleanly at handler cold-start. Install -r first for anything the
# repo pins, then force-add the specific modules we hit on import.
RUN if [ -f /opt/CosyVoice/requirements.txt ]; then \
      pip install --no-cache-dir -r /opt/CosyVoice/requirements.txt || true ; \
    fi \
 && pip install --no-cache-dir \
      hyperpyyaml==1.2.2 \
      conformer==0.3.2 \
      "wetext>=0.0.3" \
      modelscope \
      "diffusers>=0.27,<0.32" \
      lightning
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch
CMD ["python", "-u", "/app/handler.py"]
