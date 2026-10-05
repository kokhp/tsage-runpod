"""asr-align — WhisperX + WeSpeaker diarization + word-level alignment.

Hardware: A4000 (16 GB), ~4 GB VRAM, scale-to-zero.

Actions:
  transcribe(audio_url, lang_hint?, diarize=False, model="large-v3")
    → text, segments[], word_timestamps[]
  diarize(audio_url, num_speakers?)         → speakers[], segments[]
  align(audio_url, text, lang="en")         → word_segments[]
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

_whisper = {}
_diar = None
_aligners = {}


def _load_whisper(model="large-v3"):
    if model in _whisper:
        return _whisper[model]
    import whisperx  # type: ignore
    _whisper[model] = whisperx.load_model(
        model, device="cuda", compute_type="float16",
        download_root=f"{WEIGHTS_DIR}/whisperx",
    )
    return _whisper[model]


def _load_diar():
    global _diar
    if _diar is not None:
        return _diar
    import whisperx  # type: ignore
    hf = os.environ.get("HF_TOKEN")
    _diar = whisperx.DiarizationPipeline(use_auth_token=hf, device="cuda")
    return _diar


def _load_aligner(lang):
    if lang in _aligners:
        return _aligners[lang]
    import whisperx  # type: ignore
    model, meta = whisperx.load_align_model(language_code=lang, device="cuda")
    _aligners[lang] = (model, meta)
    return _aligners[lang]


def _fetch_file(url_or_b64):
    import requests
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if url_or_b64.startswith("http"):
        tmp.write(requests.get(url_or_b64, timeout=60).content)
    else:
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def action_transcribe(payload):
    import whisperx  # type: ignore
    audio_fp = _fetch_file(payload["audio_url"])
    try:
        wx = _load_whisper(payload.get("model", "large-v3"))
        audio = whisperx.load_audio(audio_fp)
        result = wx.transcribe(audio, language=payload.get("lang_hint"))
        lang = result["language"]

        align_model, meta = _load_aligner(lang)
        aligned = whisperx.align(result["segments"], align_model, meta,
                                 audio, device="cuda", return_char_alignments=False)

        if payload.get("diarize"):
            diar_pipe = _load_diar()
            diar_segs = diar_pipe(audio_fp, min_speakers=payload.get("min_speakers"),
                                  max_speakers=payload.get("max_speakers"))
            aligned = whisperx.assign_word_speakers(diar_segs, aligned)

        return {
            "language": lang,
            "segments": aligned["segments"],
            "word_segments": aligned.get("word_segments", []),
        }
    finally:
        try: os.unlink(audio_fp)
        except: pass


def action_diarize(payload):
    audio_fp = _fetch_file(payload["audio_url"])
    try:
        pipe = _load_diar()
        segs = pipe(audio_fp, min_speakers=payload.get("min_speakers"),
                    max_speakers=payload.get("max_speakers"))
        return {"segments": segs.to_dict(orient="records") if hasattr(segs, "to_dict") else list(segs)}
    finally:
        try: os.unlink(audio_fp)
        except: pass


def action_align(payload):
    import whisperx  # type: ignore
    audio_fp = _fetch_file(payload["audio_url"])
    try:
        lang = payload.get("lang", "en")
        align_model, meta = _load_aligner(lang)
        audio = whisperx.load_audio(audio_fp)
        # Build a single-segment input from the provided transcript
        segments = [{"start": 0.0, "end": len(audio)/16000, "text": payload["text"]}]
        aligned = whisperx.align(segments, align_model, meta, audio,
                                 device="cuda", return_char_alignments=False)
        return {"word_segments": aligned.get("word_segments", []),
                "segments": aligned["segments"]}
    finally:
        try: os.unlink(audio_fp)
        except: pass


ACTIONS = {"transcribe": action_transcribe, "diarize": action_diarize, "align": action_align}


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
