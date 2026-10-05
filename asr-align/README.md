# asr-align

WhisperX + WeSpeaker diarization + word-level alignment.

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~4 GB
- Scale: scale-to-zero (min 0, max 8)

## Models

| Repo | Role |
|------|------|
| WhisperX (pypi) | ASR + wav2vec2 alignment |
| WeSpeaker + pyannote.audio | diarization |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `transcribe` | `audio_url, lang_hint?, diarize=false, model="large-v3"` | `segments[], word_segments[], language` |
| `diarize` | `audio_url, min_speakers?, max_speakers?` | `segments[]` |
| `align` | `audio_url, text, lang` | `word_segments[], segments[]` |

## Env vars

`HF_TOKEN` (needed for pyannote diarization gated models), `RUNPOD_VOLUME`.
