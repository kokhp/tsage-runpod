# asr-omnilingual

Facebook Omnilingual ASR — 1600+ languages via LLM-300M (default) or CTC-7B (large).

## Hardware

- GPU: A5000 (24 GB)
- VRAM: ~5 GB for LLM-300M, ~15 GB for CTC-7B
- Scale: scale-to-zero (min 0, max 6)

## Models

| Repo | Role |
|------|------|
| `facebook/omniASR-LLM-300M` | fast omnilingual decoder |
| `facebook/omnilingual-asr-ctc-7B` | large CTC model, highest accuracy |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `transcribe` | `audio_url, lang_hint?, model="llm-300m"\|"ctc-7b"` | `text, language, duration_s` |

## Env vars

`HF_TOKEN`, `RUNPOD_VOLUME`.
