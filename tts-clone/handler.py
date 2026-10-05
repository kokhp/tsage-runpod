"""tts-clone — CosyVoice2-0.5B + OpenVoice V2 voice-clone TTS.

Hardware: A5000 (24 GB), ~6 GB VRAM steady, scale-to-zero.

Actions:
  synthesize(text, voice_id, lang="en", emotion=None, speed=1.0,
             format="wav", stream=False)       → base64 PCM/WAV
  clone(ref_audio_url, name)                   → voice_id, embedding stored on volume
  list_voices()                                → [{voice_id, name, lang, created_at}]
"""
import os
import io
import time
import json
import base64
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
VOICE_DIR = f"{VOLUME_ROOT}/voices/tts-clone"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)

COSY_REPO = "FunAudioLLM/CosyVoice2-0.5B"
OPENVOICE_REPO = "myshell-ai/OpenVoiceV2"


# ---------------------------------------------------------------------------
# Lazy model loader — modules are imported once per warm worker
# ---------------------------------------------------------------------------
_cosy = None
_ov_tone = None


def _load_cosy():
    global _cosy
    if _cosy is not None:
        return _cosy
    # cosyvoice ships as a pip package; the API wraps the CosyVoice2 class
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
    from huggingface_hub import snapshot_download

    local = snapshot_download(COSY_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    _cosy = CosyVoice2(local, load_jit=True, load_trt=False, fp16=True)
    return _cosy


def _load_openvoice():
    global _ov_tone
    if _ov_tone is not None:
        return _ov_tone
    from openvoice.api import ToneColorConverter  # type: ignore
    from huggingface_hub import snapshot_download

    local = snapshot_download(OPENVOICE_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    cfg = f"{local}/converter/config.json"
    ckpt = f"{local}/converter/checkpoint.pth"
    _ov_tone = ToneColorConverter(cfg, device="cuda")
    _ov_tone.load_ckpt(ckpt)
    return _ov_tone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_audio(url_or_b64):
    """Return mono 16k float32 numpy array from URL or base64 blob."""
    import numpy as np
    import soundfile as sf
    import librosa
    import requests

    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
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


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def action_synthesize(payload):
    import numpy as np

    text = payload["text"]
    voice_id = payload.get("voice_id", "default")
    lang = payload.get("lang", "en")
    speed = float(payload.get("speed", 1.0))
    emotion = payload.get("emotion")

    cosy = _load_cosy()

    voice_fp = Path(VOICE_DIR) / f"{voice_id}.wav"
    if not voice_fp.exists() and voice_id != "default":
        return {"error": f"voice_id {voice_id} not found; call clone first"}

    if voice_id == "default":
        # CosyVoice2 bundles pretrained speakers keyed by lang
        chunks = []
        for ch in cosy.inference_sft(text, lang, speed=speed):
            chunks.append(ch["tts_speech"].numpy().squeeze())
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    else:
        import soundfile as sf
        ref_audio, _ = sf.read(str(voice_fp), dtype="float32")
        chunks = []
        for ch in cosy.inference_zero_shot(text, "", ref_audio, speed=speed):
            chunks.append(ch["tts_speech"].numpy().squeeze())
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    return {
        "audio_b64": _pack_wav(audio, 22050),
        "sample_rate": 22050,
        "voice_id": voice_id,
        "lang": lang,
        "duration_s": len(audio) / 22050,
    }


def action_clone(payload):
    import soundfile as sf

    name = payload.get("name") or f"vc_{int(time.time())}"
    ref = _fetch_audio(payload["ref_audio_url"])
    out_fp = Path(VOICE_DIR) / f"{name}.wav"
    sf.write(str(out_fp), ref, 16000, format="WAV", subtype="PCM_16")

    # Pre-compute OpenVoice tone embedding for faster synth later
    try:
        from openvoice import se_extractor  # type: ignore
        ov = _load_openvoice()
        tgt_se, _ = se_extractor.get_se(str(out_fp), ov, vad=True)
        emb_fp = Path(VOICE_DIR) / f"{name}.se.npy"
        import numpy as np
        np.save(str(emb_fp), tgt_se.cpu().numpy())
    except Exception as e:
        return {"voice_id": name, "warning": f"openvoice embed failed: {e}"}

    return {"voice_id": name, "ref_path": str(out_fp)}


def action_list_voices(payload):
    out = []
    for fp in sorted(Path(VOICE_DIR).glob("*.wav")):
        out.append({
            "voice_id": fp.stem,
            "ref_path": str(fp),
            "created_at": fp.stat().st_mtime,
        })
    return {"voices": out}


# ---------------------------------------------------------------------------
# Handler entry
# ---------------------------------------------------------------------------
ACTIONS = {
    "synthesize": action_synthesize,
    "clone": action_clone,
    "list_voices": action_list_voices,
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
