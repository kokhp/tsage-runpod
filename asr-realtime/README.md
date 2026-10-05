# asr-realtime

NVIDIA Parakeet-TDT 0.6B v3 — streaming-friendly English ASR.

## Hardware

- GPU: A4000 (16 GB)
- VRAM: ~3 GB
- Scale: min 1 / max 20 (keeps a warm worker; essential for sub-100 ms partial latency)

## Models

| Repo | Role |
|------|------|
| `nvidia/parakeet-tdt-0.6b-v3` | 0.6 B param TDT ASR, best WER/latency on English streaming |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `transcribe` | `audio_url, timestamps=true` | `text, segments[]` |
| `transcribe_chunk` | `audio_b64, state_id?, final=false` | `text, state_id, final` |

Clients drive a stream by posting consecutive `transcribe_chunk` requests with the same `state_id`, then one more with `final=true` to flush.

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
