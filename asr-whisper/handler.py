"""asr-whisper — Whisper-Large-v3 + turbo + Faster-Whisper.

Hardware: A5000 (24 GB), ~5 GB VRAM, scale-to-zero.

Actions:
  transcribe(audio_url, lang_hint?, timestamps=True, model="turbo"|"large-v3"|"faster-large-v3",
             diarize=False, format="json") → segments[], text, language
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

_models = {}


def _load(name):
    if name in _models:
        return _models[name]
    import torch
    if name.startswith("faster-"):
        from faster_whisper import WhisperModel  # type: ignore
        size = name.replace("faster-", "")
        _models[name] = WhisperModel(size, device="cuda", compute_type="float16",
                                     download_root=f"{WEIGHTS_DIR}/faster-whisper")
    else:
        from transformers import pipeline
        repo = {
            "large-v3": "openai/whisper-large-v3",
            "turbo": "openai/whisper-large-v3-turbo",
        }.get(name, f"openai/{name}")
        _models[name] = pipeline(
            "automatic-speech-recognition",
            model=repo,
            torch_dtype=torch.float16,
            device="cuda",
            return_timestamps=True,
            chunk_length_s=30,
            model_kwargs={"cache_dir": WEIGHTS_DIR, "token": os.environ.get("HF_TOKEN")},
        )
    return _models[name]


def _fetch_file(url_or_b64):
    import requests
    suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if url_or_b64.startswith("http"):
        tmp.write(requests.get(url_or_b64, timeout=60).content)
    else:
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def action_transcribe(payload):
    audio = _fetch_file(payload["audio_url"])
    lang = payload.get("lang_hint")
    model_name = payload.get("model", "turbo")
    timestamps = bool(payload.get("timestamps", True))

    try:
        model = _load(model_name)
        if model_name.startswith("faster-"):
            segments, info = model.transcribe(
                audio, language=lang, word_timestamps=timestamps,
                vad_filter=True,
            )
            segs = [{
                "start": s.start, "end": s.end, "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.word}
                          for w in (s.words or [])] if timestamps else None,
            } for s in segments]
            return {
                "segments": segs,
                "text": " ".join(s["text"].strip() for s in segs).strip(),
                "language": info.language,
                "duration": info.duration,
                "model": model_name,
            }
        else:
            out = model(audio, generate_kwargs={"language": lang} if lang else None,
                        return_timestamps=timestamps)
            return {
                "text": out["text"],
                "segments": out.get("chunks", []),
                "language": lang or "auto",
                "model": model_name,
            }
    finally:
        try: os.unlink(audio)
        except: pass


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
