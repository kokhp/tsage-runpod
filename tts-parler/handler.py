"""tts-parler — Parler-TTS Large v1 descriptive-prompt TTS.

Hardware: A4000 (16 GB), ~5 GB VRAM, scale-to-zero.

Actions:
  synthesize(text, description, lang="en", speed=1.0, format="wav") → base64 WAV
"""
import os
import io
import base64
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

PARLER_REPO = "parler-tts/parler-tts-large-v1"

_parler = None


def _load():
    global _parler
    if _parler is not None:
        return _parler
    import torch
    from huggingface_hub import snapshot_download
    local = snapshot_download(PARLER_REPO, cache_dir=WEIGHTS_DIR,
                              token=os.environ.get("HF_TOKEN"))
    from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(local).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(local)
    _parler = (model, tok)
    return _parler


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_synthesize(payload):
    import torch
    model, tok = _load()
    text = payload["text"]
    description = payload.get(
        "description",
        "A calm, clear voice speaks naturally at a moderate pace with high recording quality.",
    )
    input_ids = tok(description, return_tensors="pt").input_ids.to("cuda")
    prompt_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        audio = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
    audio_np = audio.cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    return {
        "audio_b64": _pack_wav(audio_np, sr),
        "sample_rate": int(sr),
        "duration_s": len(audio_np) / sr,
    }


ACTIONS = {"synthesize": action_synthesize}


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
