# dialogue-en

Nari Labs Dia-1.6B — English multi-speaker dialogue TTS with `[S1]`/`[S2]` turn markers.

## Hardware

- GPU: RTX 4090 (24 GB)
- VRAM: ~16 GB
- Scale: scale-to-zero (min 0, max 3)
- English only — do not route non-English prompts here.

## Models

| Repo | Role |
|------|------|
| `nari-labs/Dia-1.6B-0626` | 1.6 B dialogue TTS |
| `FunAudioLLM/CosyVoice2-0.5B` (optional ref) | voice-prompt audio for stable casting |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `generate` | `script, voices?=[voice_ids], cfg_scale=3.0, seed?` | `audio_b64, sample_rate=44100, duration_s` |

Script format: `[S1] ... \n[S2] ... \n[S1] ...`.

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
