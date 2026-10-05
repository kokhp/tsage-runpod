# separator-cpu

Demucs `htdemucs_ft` on CPU — cheap 4-stem separation (vocals/drums/bass/other).

## Hardware

- CPU: 8-vCPU (RunPod CPU3-8)
- Memory: 32 GB
- Scale: 24/7 pod (min 1, max 1)
- Not scale-to-zero: cold-start load is ~60 s.

## Models

| Source | Role |
|--------|------|
| `demucs` pypi `htdemucs_ft` | 4-stem fine-tuned Demucs HT |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `separate` | `audio_url, mode="vocals"\|"me"\|"both"\|"4stem"` | per-stem base64 WAVs + `sample_rate` |

## Env vars

`RUNPOD_VOLUME`.
