# tts-cpu

MeloTTS CPU container — cheap fallback for low-stakes synthesis.

## Hardware

- CPU: 4-vCPU (RunPod CPU3-4)
- Memory: 16 GB
- Scale: 24/7 pod (min 1, max 1)
- Not scale-to-zero: cold-start load is ~45 s so a floor of one keeps latency reasonable.

## Models

| Repo | Role |
|------|------|
| `myshell-ai/MeloTTS-English`, `-Spanish`, `-French`, `-Chinese`, `-Japanese`, `-Korean` | CPU-friendly multi-lang TTS |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, lang="EN", speaker_id?, speed=1.0` | `audio_b64, sample_rate, speaker_id` |
| `list_voices` | `lang="EN"` | `voices[]` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
