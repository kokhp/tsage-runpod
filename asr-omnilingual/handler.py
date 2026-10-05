"""asr-omnilingual — Facebook Omnilingual ASR (LLM-300M + CTC-7B, 1600+ langs).

Hardware: A5000 (24 GB), ~15 GB VRAM, scale-to-zero.

Actions:
  transcribe(audio_url, lang_hint?, model="llm-300m"|"ctc-7b", timestamps=True)
    → text, segments[], language
"""
import os
import io
import base64
import tempfile
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

LLM_REPO = "facebook/omniASR-LLM-300M"
CTC_REPO = "facebook/omnilingual-asr-ctc-7B"

_models = {}


def _load(variant):
    if variant in _models:
        return _models[variant]
    import torch
    from transformers import AutoModel, AutoProcessor
    repo = LLM_REPO if variant == "llm-300m" else CTC_REPO
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    proc = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = AutoModel.from_pretrained(local, trust_remote_code=True,
                                      torch_dtype=torch.float16).to("cuda").eval()
    _models[variant] = (model, proc)
    return _models[variant]


def _fetch_audio(url_or_b64):
    import numpy as np, soundfile as sf, librosa, requests
    if url_or_b64.startswith("http"):
        data = requests.get(url_or_b64, timeout=60).content
    else:
        data = base64.b64decode(url_or_b64)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def action_transcribe(payload):
    import torch
    audio = _fetch_audio(payload["audio_url"])
    lang = payload.get("lang_hint")
    variant = payload.get("model", "llm-300m")
    model, proc = _load(variant)

    inputs = proc(audio=audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}
    with torch.no_grad():
        if hasattr(model, "generate"):
            out_ids = model.generate(**inputs, max_new_tokens=512,
                                     language=lang) if lang else model.generate(**inputs, max_new_tokens=512)
        else:
            logits = model(**inputs).logits
            out_ids = logits.argmax(-1)
    text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
    return {
        "text": text,
        "language": lang or "auto",
        "model": variant,
        "duration_s": len(audio) / 16000,
    }


ACTIONS = {"transcribe": action_transcribe}


def handler(job):
    payload = job.get("input") or {}
    action = payload.get("action")
    fn = ACTIONS.get(action)
    if not fn:
        return {"error": f"unknown action: {action}", "valid": list(ACTIONS)}
    try:
        return fn(payload)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
