# separator-gpu

Fast vocal / music-element separation with BS-RoFormer.

## Hardware

- GPU: RTX 4090 (24 GB)
- VRAM: ~8 GB
- Scale: scale-to-zero (min 0, max 6)

## Models

| Source | Role |
|--------|------|
| `lucidrains/BS-RoFormer` (code) + `KimberleyJensen/BS-Roformer` (weights) | SOTA vocal separation at ~9.6 SDR |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `separate` | `audio_url, mode="vocals"\|"me"\|"both"` | `vocals_b64?, instrumental_b64?, sample_rate` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
