# tts-design

Emotive / descriptive-prompt TTS: Maya1 3B and VoxCPM2.

## Hardware

- GPU: A5000 (24 GB)
- Steady VRAM: ~12 GB
- Scale: scale-to-zero (min 0, max 8)

## Models

| Repo | Role |
|------|------|
| `maya-research/maya1` | 3 B emotion-token LLM TTS (SNAC-24k decoder) |
| `openbmb/VoxCPM2` | descriptive-prompt design TTS |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, model="maya1"\|"voxcpm2", style?, emotion?, lang?, speed?` | `audio_b64, sample_rate=24000, duration_s` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
