"""dialogue-en — nari-labs/Dia-1.6B-0626 English multi-speaker dialogue TTS.

Hardware: RTX 4090 (24 GB), ~16 GB VRAM, scale-to-zero (English only).

Actions:
  generate(script, voices=[voice_ids], cfg_scale=3.0, seed?, format="wav")
    → audio_b64, sample_rate, duration_s
    script format:
      "[S1] Hello.\n[S2] Hi there, how are you?\n[S1] Good, thanks."
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
VOICE_DIR = f"{VOLUME_ROOT}/voices/dialogue-en"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)

DIA_REPO = "nari-labs/Dia-1.6B-0626"

_dia = None


def _load():
    global _dia
    if _dia is not None:
        return _dia
    import torch
    from dia.model import Dia  # type: ignore
    from huggingface_hub import snapshot_download
    local = snapshot_download(DIA_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    _dia = Dia.from_local(local, compute_dtype=torch.float16, device="cuda")
    return _dia


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_generate(payload):
    dia = _load()
    script = payload["script"]
    cfg_scale = float(payload.get("cfg_scale", 3.0))
    seed = payload.get("seed")

    audio_prompts = []
    for vid in payload.get("voices") or []:
        fp = Path(VOICE_DIR) / f"{vid}.wav"
        if fp.exists():
            audio_prompts.append(str(fp))

    kwargs = {"cfg_scale": cfg_scale, "temperature": 1.3, "top_p": 0.95}
    if audio_prompts:
        kwargs["audio_prompt"] = audio_prompts[0]  # Dia takes one reference tape
    if seed is not None:
        import torch
        torch.manual_seed(int(seed))

    audio = dia.generate(script, **kwargs)
    sr = 44100
    return {
        "audio_b64": _pack_wav(audio, sr),
        "sample_rate": sr,
        "duration_s": len(audio) / sr,
    }


ACTIONS = {"generate": action_generate}


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
