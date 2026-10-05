# asr-whisper

Whisper family: Whisper-Large-v3, Whisper-Large-v3-turbo, Faster-Whisper (ctranslate2).

## Hardware

- GPU: A5000 (24 GB)
- VRAM: ~5 GB (one model loaded; multi-model if memory allows)
- Scale: scale-to-zero (min 0, max 15)

## Models

| Repo | Role |
|------|------|
| `openai/whisper-large-v3` | top accuracy, slower |
| `openai/whisper-large-v3-turbo` | 8x faster, slightly lower accuracy |
| Faster-Whisper via `ctranslate2` | int8 quantised path |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `transcribe` | `audio_url, lang_hint?, timestamps=true, model="turbo"\|"large-v3"\|"faster-large-v3"` | `text, segments[], language, duration` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
