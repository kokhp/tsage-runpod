# tts-parler

Descriptive-prompt TTS using Parler-TTS Large v1.

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~5 GB
- Scale: scale-to-zero (min 0, max 8)

## Models

| Repo | Role |
|------|------|
| `parler-tts/parler-tts-large-v1` | 2.3 B English TTS with natural-language style description |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, description?, lang?, speed?` | `audio_b64, sample_rate, duration_s` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
