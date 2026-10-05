FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg git git-lfs libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# openai-whisper 20231117 (and other legacy pkgs) use pkg_resources which
# was removed from setuptools >= 70. Pin + add wheel_stub for nvidia stubs.
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir 'setuptools<70' wheel_stub hf_transfer
RUN pip install --no-cache-dir -r /app/requirements.txt

# CosyVoice repo has no proper setup.py; the `cosyvoice` package on PyPI is an
# unrelated project and does NOT provide `cosyvoice.cli`. Clone the real repo
# and expose it (plus its Matcha-TTS submodule) on PYTHONPATH so the handler's
# `from cosyvoice.cli.cosyvoice import CosyVoice3` import resolves.
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
 && git -C /opt/CosyVoice submodule update --init --recursive || true
# CosyVoice's own import chain (cosyvoice.cli.cosyvoice -> cli.frontend etc)
# pulls in whisper, inflect, HyperPyYAML, conformer, modelscope, diffusers,
# lightning, onnx, pyworld and friends. The repo's requirements.txt also
# pins deepspeed/tensorrt/tensorboard/gradio and clobbers torch 2.4.1 with
# 2.3.1, so we do NOT run `pip install -r`. Instead install only the modules
# that are actually imported at handler cold-start, and keep torch pinned
# at the base image version.
RUN pip install --no-cache-dir \
      openai-whisper==20231117 \
      HyperPyYAML==1.2.3 \
      conformer==0.3.2 \
      inflect==7.3.1 \
      "modelscope>=1.15,<1.21" \
      "diffusers==0.29.0" \
      "lightning==2.2.4" \
      "transformers==4.46.3" \
      "onnx>=1.14,<2" \
      "onnxruntime-gpu==1.18.0" \
      "pyworld>=0.3,<0.4" \
      "x-transformers>=1.30,<3" \
      "wetext>=0.0.3" \
      "networkx>=3.1,<4" \
      "gdown>=5.1,<6" \
      "pyarrow>=15,<19"
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch
CMD ["python", "-u", "/app/handler.py"]
