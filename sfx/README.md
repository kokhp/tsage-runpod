# sfx

Riffusion v1 — text-prompt SFX / short-music generation via spectrogram-image diffusion + Griffin-Lim.

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~5 GB
- Scale: scale-to-zero (min 0, max 4)

## Models

| Repo | Role |
|------|------|
| `riffusion/riffusion-model-v1` | Stable-Diffusion-1.5 fine-tune emitting mel-spectrogram images |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `generate` | `prompt, duration_s=5, seed?` | `audio_b64, spectrogram_png_b64, sample_rate=22050` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
