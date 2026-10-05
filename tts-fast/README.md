# tts-fast

Low-latency TTS: Qwen3-TTS-12Hz-0.6B-Base (default) and Kokoro-82M (fallback / multilang).

## Hardware

- GPU: A4000 (16 GB)
- Steady VRAM: ~4 GB
- Scale mode: scale-to-zero (min 0, max 10)

## Models

| Repo | Role | Pin |
|------|------|-----|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 12 Hz Qwen-tier fast TTS | main |
| `hexgrad/Kokoro-82M` | 82 M-param rapid TTS, multi-voice | main |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, voice_id="af_heart", lang="en", speed=1.0, model="kokoro"\|"qwen"` | `audio_b64, sample_rate=24000, duration_s` |
| `list_voices` | `model="kokoro"` | `voices[]` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
