"""asr-realtime — NVIDIA Parakeet-TDT 0.6B v3 streaming ASR.

Hardware: A4000 (16 GB), ~3 GB VRAM, min-worker 1 to keep sub-100ms warm latency.

Actions:
  transcribe(audio_url, lang_hint?, timestamps=True)              → text, segments
  transcribe_chunk(audio_b64, state_id?, final=False)             → partial text, state_id
    - caller maintains state_id across chunks for streaming decode
"""
import os
import io
import time
import uuid
import base64
import tempfile
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

PARAKEET_REPO = "nvidia/parakeet-tdt-0.6b-v3"

_model = None
_streams = {}


def _load():
    global _model
    if _model is not None:
        return _model
    import nemo.collections.asr as nemo_asr  # type: ignore
    _model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_REPO)
    _model.eval()
    _model = _model.cuda()
    return _model


def _fetch_file(url_or_b64):
    """Download/decode and normalise to a 16 kHz mono WAV.

    NeMo parakeet-tdt expects (batch, time); multi-channel input raises
    "Input shape expected = (batch, time) | Input shape found : (1, 2, N)".
    Always downmix to mono + resample to 16 kHz before handing NeMo a path.
    """
    import requests, soundfile as sf, numpy as np
    raw = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    if url_or_b64.startswith("http"):
        raw.write(requests.get(url_or_b64, timeout=60).content)
    else:
        raw.write(base64.b64decode(url_or_b64))
    raw.close()
    audio, sr = sf.read(raw.name, dtype="float32", always_2d=False)
    try:
        os.unlink(raw.name)
    except Exception:
        pass
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    sf.write(out.name, audio.astype(np.float32), sr, subtype="PCM_16")
    return out.name


def action_transcribe(payload):
    model = _load()
    audio = _fetch_file(payload["audio_url"])
    try:
        hyps = model.transcribe([audio], timestamps=bool(payload.get("timestamps", True)))
        if hyps and hasattr(hyps[0], "text"):
            text = hyps[0].text
            segs = getattr(hyps[0], "timestamp", {}).get("segment", []) if hasattr(hyps[0], "timestamp") else []
        else:
            text = hyps[0] if hyps else ""
            segs = []
        return {"text": text, "segments": segs, "model": "parakeet-tdt-0.6b-v3"}
    finally:
        try: os.unlink(audio)
        except: pass


def action_transcribe_chunk(payload):
    """Minimal buffered streaming: append chunk to state buffer, re-decode on final."""
    import numpy as np, soundfile as sf
    model = _load()
    state_id = payload.get("state_id") or str(uuid.uuid4())
    state = _streams.setdefault(state_id, {"buffer": [], "created": time.time()})

    chunk = base64.b64decode(payload["audio_b64"])
    audio, sr = sf.read(io.BytesIO(chunk), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    state["buffer"].append(audio)

    final = bool(payload.get("final"))
    if final:
        full = np.concatenate(state["buffer"]).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, full, 16000)
        tmp.close()
        try:
            hyps = model.transcribe([tmp.name], timestamps=True)
            text = hyps[0].text if hasattr(hyps[0], "text") else hyps[0]
        finally:
            os.unlink(tmp.name)
            _streams.pop(state_id, None)
        return {"state_id": state_id, "text": text, "final": True}

    # Partial — decode the running buffer every call (cheap on parakeet 0.6B)
    partial = np.concatenate(state["buffer"][-20:]).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, partial, 16000)
    tmp.close()
    try:
        hyps = model.transcribe([tmp.name], timestamps=False)
        text = hyps[0].text if hasattr(hyps[0], "text") else hyps[0]
    finally:
        os.unlink(tmp.name)
    return {"state_id": state_id, "text": text, "final": False}


ACTIONS = {
    "transcribe": action_transcribe,
    "transcribe_chunk": action_transcribe_chunk,
}


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
