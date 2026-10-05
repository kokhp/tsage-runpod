# vc

Voice conversion: kNN-VC (default) and FreeVC (fallback).

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~6 GB
- Scale: scale-to-zero (min 0, max 6)

## Models

| Source | Role |
|--------|------|
| `bshall/knn-vc` (torch.hub) | kNN voice conversion over WavLM features |
| `OlaWod/FreeVC` | end-to-end any-to-any voice conversion |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `convert` | `source_audio_url, (target_voice_id\|target_audio_url), model, strength` | `audio_b64, sample_rate=16000` |
| `register` | `voice_id, ref_audio_url` | `voice_id, ref_path` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
