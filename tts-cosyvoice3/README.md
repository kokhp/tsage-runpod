# tts-cosyvoice3

Fun-CosyVoice3-0.5B-2512 — newest FunAudioLLM TTS, slightly higher quality than CosyVoice2, same API shape.

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~4 GB
- Scale: scale-to-zero (min 0, max 8)

## Models

| Repo | Role |
|------|------|
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | zero-shot clone + base TTS |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, voice_id?, lang="en", speed=1.0` | `audio_b64, sample_rate=24000` |
| `clone` | `ref_audio_url, name?` | `voice_id, ref_path` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
