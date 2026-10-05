"""tts-fast — low-latency TTS: Qwen3-TTS-12Hz-0.6B + Kokoro-82M.

Hardware: A4000 (16 GB), ~4 GB VRAM, scale-to-zero.

Actions:
  synthesize(text, voice_id="af_heart", lang="en", speed=1.0, model="kokoro",
             format="wav", stream=False)   → base64 WAV
  list_voices(model?)                      → list of built-in voices
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

QWEN_TTS_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
KOKORO_REPO = "hexgrad/Kokoro-82M"


# ---------------------------------------------------------------------------
_kokoro = None
_qwen = None


def _load_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    from kokoro import KPipeline  # type: ignore
    _kokoro = KPipeline(lang_code="a")  # 'a' = en-US default; swapped per request
    return _kokoro


def _load_qwen_tts():
    global _qwen
    if _qwen is not None:
        return _qwen
    from huggingface_hub import snapshot_download
    import torch
    local = snapshot_download(QWEN_TTS_REPO, cache_dir=WEIGHTS_DIR,
                              token=os.environ.get("HF_TOKEN"))
    # Qwen3-TTS exposes a lightweight AutoModel API
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = AutoModel.from_pretrained(local, trust_remote_code=True,
                                      torch_dtype=torch.float16).to("cuda").eval()
    _qwen = (model, proc)
    return _qwen


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
LANG_CODE_MAP = {"en": "a", "en-gb": "b", "es": "e", "fr": "f", "hi": "h",
                 "it": "i", "ja": "j", "pt": "p", "zh": "z"}


def action_synthesize(payload):
    import numpy as np

    text = payload["text"]
    voice_id = payload.get("voice_id", "af_heart")
    lang = payload.get("lang", "en")
    speed = float(payload.get("speed", 1.0))
    model_name = payload.get("model", "kokoro")

    if model_name == "qwen":
        model, proc = _load_qwen_tts()
        import torch
        inputs = proc(text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, speaker=voice_id)
        audio = out.audio.squeeze().cpu().float().numpy()
        sr = 24000
    else:  # kokoro
        pipe = _load_kokoro()
        pipe.lang_code = LANG_CODE_MAP.get(lang, "a")
        chunks = []
        for _, _, audio_chunk in pipe(text, voice=voice_id, speed=speed):
            chunks.append(audio_chunk)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        sr = 24000

    return {
        "audio_b64": _pack_wav(audio, sr),
        "sample_rate": sr,
        "voice_id": voice_id,
        "lang": lang,
        "model": model_name,
        "duration_s": len(audio) / sr,
    }


def action_list_voices(payload):
    model_name = payload.get("model", "kokoro")
    if model_name == "kokoro":
        # Kokoro ships bundled voices; return the canonical set
        return {"voices": [
            {"voice_id": "af_heart", "lang": "en", "gender": "female"},
            {"voice_id": "af_bella", "lang": "en", "gender": "female"},
            {"voice_id": "am_michael", "lang": "en", "gender": "male"},
            {"voice_id": "bf_emma", "lang": "en-gb", "gender": "female"},
            {"voice_id": "bm_george", "lang": "en-gb", "gender": "male"},
            {"voice_id": "jf_alpha", "lang": "ja", "gender": "female"},
            {"voice_id": "zf_xiaobei", "lang": "zh", "gender": "female"},
        ]}
    return {"voices": []}


ACTIONS = {"synthesize": action_synthesize, "list_voices": action_list_voices}


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
