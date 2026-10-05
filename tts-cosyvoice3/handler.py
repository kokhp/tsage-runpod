"""tts-cosyvoice3 — Fun-CosyVoice3-0.5B-2512.

Hardware: A4000 (16 GB), ~4 GB VRAM, scale-to-zero.

Actions:
  synthesize(text, voice_id?, lang="en", speed=1.0, format="wav") → base64 WAV
  clone(ref_audio_url, name) → voice_id
"""
import os
import io
import time
import base64
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
VOICE_DIR = f"{VOLUME_ROOT}/voices/tts-cosyvoice3"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)

COSY3_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    from huggingface_hub import snapshot_download
    local = snapshot_download(COSY3_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore
        _model = CosyVoice3(local, load_jit=True, fp16=True)
    except Exception:
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        _model = CosyVoice2(local, load_jit=True, fp16=True)
    return _model


def _fetch_audio(url_or_b64):
    import numpy as np, soundfile as sf, librosa, requests
    if url_or_b64.startswith("http"):
        data = requests.get(url_or_b64, timeout=30).content
    else:
        data = base64.b64decode(url_or_b64)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_synthesize(payload):
    import numpy as np
    cosy = _load()
    text = payload["text"]
    voice_id = payload.get("voice_id", "default")
    lang = payload.get("lang", "en")
    speed = float(payload.get("speed", 1.0))

    voice_fp = Path(VOICE_DIR) / f"{voice_id}.wav"
    if voice_id != "default" and not voice_fp.exists():
        return {"error": f"voice_id {voice_id} not found; call clone first"}

    chunks = []
    if voice_id == "default":
        for ch in cosy.inference_sft(text, lang, speed=speed):
            chunks.append(ch["tts_speech"].numpy().squeeze())
    else:
        import soundfile as sf
        ref, _ = sf.read(str(voice_fp), dtype="float32")
        for ch in cosy.inference_zero_shot(text, "", ref, speed=speed):
            chunks.append(ch["tts_speech"].numpy().squeeze())
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sr = 24000
    return {"audio_b64": _pack_wav(audio, sr), "sample_rate": sr,
            "voice_id": voice_id, "duration_s": len(audio) / sr}


def action_clone(payload):
    import soundfile as sf
    name = payload.get("name") or f"vc_{int(time.time())}"
    ref = _fetch_audio(payload["ref_audio_url"])
    out_fp = Path(VOICE_DIR) / f"{name}.wav"
    sf.write(str(out_fp), ref, 16000, format="WAV", subtype="PCM_16")
    return {"voice_id": name, "ref_path": str(out_fp)}


ACTIONS = {"synthesize": action_synthesize, "clone": action_clone}


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
