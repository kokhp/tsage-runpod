"""separator-cpu — Demucs htdemucs_ft on CPU, 24/7 pod.

Actions:
  separate(audio_url, mode="vocals"|"me"|"both"|"4stem")
    → {vocals_b64?, drums_b64?, bass_b64?, other_b64?, instrumental_b64?, sample_rate}
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

_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    from demucs.pretrained import get_model  # type: ignore
    _model = get_model("htdemucs_ft")
    _model.cpu().eval()
    return _model


def _fetch_file(url_or_b64, suffix=".wav"):
    import requests
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if url_or_b64.startswith("http"):
        tmp.write(requests.get(url_or_b64, timeout=120).content)
    else:
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio.T if audio.ndim == 2 else audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_separate(payload):
    import torch, torchaudio
    from demucs.apply import apply_model  # type: ignore
    model = _load()
    mode = payload.get("mode", "vocals")
    audio_fp = _fetch_file(payload["audio_url"])
    try:
        wav, sr = torchaudio.load(audio_fp)
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0), split=True, overlap=0.1)[0]
        # order: drums, bass, other, vocals
        drums, bass, other, vocals = sources
        out = {"sample_rate": sr}
        if mode in ("vocals", "both"):
            out["vocals_b64"] = _pack_wav(vocals.numpy(), sr)
        if mode in ("me", "both", "instrumental"):
            instrumental = drums + bass + other
            out["instrumental_b64"] = _pack_wav(instrumental.numpy(), sr)
        if mode == "4stem":
            out["vocals_b64"] = _pack_wav(vocals.numpy(), sr)
            out["drums_b64"] = _pack_wav(drums.numpy(), sr)
            out["bass_b64"] = _pack_wav(bass.numpy(), sr)
            out["other_b64"] = _pack_wav(other.numpy(), sr)
        return out
    finally:
        try: os.unlink(audio_fp)
        except: pass


ACTIONS = {"separate": action_separate}


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
