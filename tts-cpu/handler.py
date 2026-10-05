"""tts-cpu — MeloTTS on CPU (4 vCPU, 24/7 pod).

Actions:
  synthesize(text, lang="EN", speaker_id?, speed=1.0, format="wav") → base64 WAV
  list_voices() → built-in MeloTTS speakers per language
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

_models = {}


def _load(lang):
    if lang in _models:
        return _models[lang]
    from melo.api import TTS  # type: ignore
    model = TTS(language=lang, device="cpu")
    _models[lang] = model
    return model


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_synthesize(payload):
    text = payload["text"]
    lang = payload.get("lang", "EN").upper()
    speaker_id = payload.get("speaker_id") or payload.get("voice_id")
    speed = float(payload.get("speed", 1.0))

    model = _load(lang)
    speakers = model.hps.data.spk2id
    if not speaker_id:
        speaker_id = next(iter(speakers))
    spk = speakers.get(speaker_id)
    if spk is None:
        return {"error": f"unknown speaker_id {speaker_id}", "valid": list(speakers)}

    buf = io.BytesIO()
    model.tts_to_file(text, spk, buf, speed=speed, quiet=True, format="wav")
    buf.seek(0)
    return {
        "audio_b64": base64.b64encode(buf.read()).decode(),
        "sample_rate": model.hps.data.sampling_rate,
        "lang": lang,
        "speaker_id": speaker_id,
    }


def action_list_voices(payload):
    lang = payload.get("lang", "EN").upper()
    try:
        model = _load(lang)
        return {"lang": lang, "voices": list(model.hps.data.spk2id)}
    except Exception as e:
        return {"error": str(e)}


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
