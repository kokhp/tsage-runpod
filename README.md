# tsage-runpod — RunPod Serverless Container Fleet

Monorepo of 19 RunPod Serverless container images backing `api.translatorsage.com`.

Scaffolded 2026-10-05 per Section 4 of `translatorsage-engine/docs/launch-plan.md`.

## Containers

| # | Name | Role | Primary models | GPU tier | VRAM | Scale mode |
|---|------|------|----------------|----------|------|------------|
| 1 | `tts-clone` | voice-clone TTS | CosyVoice2-0.5B + OpenVoice V2 | A5000 | ~6 GB | scale-to-zero |
| 2 | `tts-fast` | low-latency TTS | Qwen3-TTS-12Hz-0.6B + Kokoro-82M | A4000 | ~4 GB | scale-to-zero |
| 3 | `tts-design` | emotive design TTS | Maya1 3B + VoxCPM2 | A5000 | ~12 GB | scale-to-zero |
| 4 | `tts-parler` | descriptive-prompt TTS | Parler-TTS Large v1 | A4000 | ~5 GB | scale-to-zero |
| 5 | `tts-cosyvoice3` | newest Fun-CosyVoice | Fun-CosyVoice3-0.5B-2512 | A4000 | ~4 GB | scale-to-zero |
| 6 | `tts-cpu` | cheap backup TTS | MeloTTS | CPU 4-vCPU | n/a | 24/7 pod |
| 7 | `asr-whisper` | Whisper family | Whisper-Large-v3 + turbo + Faster-Whisper | A5000 | ~5 GB | scale-to-zero |
| 8 | `asr-omnilingual` | 1600-lang ASR | facebook/omniASR-LLM-300M + CTC-7B | A5000 | ~15 GB | scale-to-zero |
| 9 | `asr-realtime` | streaming ASR | NVIDIA Parakeet-TDT 0.6B v3 | A4000 | ~3 GB | min-worker 1 |
| 10 | `asr-align` | diarize + align | WhisperX + WeSpeaker | A4000 | ~4 GB | scale-to-zero |
| 11 | `separator-gpu` | fast vocal separation | BS-RoFormer | RTX 4090 | ~8 GB | scale-to-zero |
| 12 | `separator-cpu` | cheap separation | Demucs htdemucs_ft | CPU 8-vCPU | n/a | 24/7 pod |
| 13 | `sfx` | SFX / foley gen | Riffusion v1 | A4000 | ~5 GB | scale-to-zero |
| 14 | `dialogue-en` | English multi-speaker | Dia-1.6B-0626 + CosyVoice2 ref | RTX 4090 | ~16 GB | scale-to-zero |
| 15 | `vc` | voice conversion | kNN-VC + FreeVC | A4000 | ~6 GB | scale-to-zero |
| 16 | `mt` | translation | madlad400-10b-mt GPTQ-int4 + m2m100 | A5000 | ~8 GB | scale-to-zero |
| 17 | `agents-llm` | router / tool-call LLM | Qwen3-14B int8 + Mistral-Nemo-12B | A5000 | ~15 GB | min-worker 1 |
| 18 | `agents-llm-big` | burst reasoning | Qwen2.5-32B-Instruct int8 | A100 80 GB | ~35 GB | scale-to-zero |
| 19 | `lora-train` | per-user LoRA fine-tune | CosyVoice2 PVC pipeline | L40S 48 GB | n/a | user-triggered |

## Container layout

Every subdirectory follows the same shape:

```
<name>/
  Dockerfile          # base image + system/python deps
  handler.py          # runpod.serverless.start handler with actions dict
  requirements.txt    # pinned Python deps
  README.md           # model SHAs, VRAM, endpoint actions, env vars
  .runpodignore       # files to exclude from the build context
  deploy.sh           # build + push + create/update serverless endpoint
```

## Common handler contract

Every handler accepts a job of shape:

```json
{"input": {"action": "<verb>", "...": "..."}}
```

Returns either `{"<keys>": "..."}` on success or `{"error": "...", "trace": "..."}` on failure. Audio I/O is base64 PCM/WAV in and base64 PCM/WAV out unless explicitly URL based.

Actions per capability:

- **TTS**: `synthesize(text, voice_id, lang, emotion, speed, format, stream)`; clone-capable adds `clone(ref_audio_url, name)`.
- **ASR**: `transcribe(audio_url, lang_hint, diarize, timestamps, model)`; realtime adds `transcribe_stream(ws)`.
- **Separator**: `separate(audio_url, mode=vocals|me|both)`.
- **MT**: `translate(text, src_lang, tgt_lang, register)`.
- **LLM**: `chat(messages, tools, max_tokens, temperature, stream)`.
- **LoRA train**: `train(voice_id, ref_audio_url, user_id)` returns `job_id`, poll via `status(job_id)`.
- **Diarize/Align**: `diarize(audio_url)`, `align(audio_url, text)`.
- **SFX**: `generate(prompt, duration_s)`.
- **Dialogue**: `generate(script, voices=[voice_ids])`.

## Credentials

Deploy scripts expect the following env vars (or source via dotenv files from `~/.claude/env/`):

- `RUNPOD_API_KEY` — scope: full account, from `~/.claude/env/runpod.env`
- `HF_TOKEN` — HuggingFace read token, from `~/.claude/env/huggingface.env`
- `REGISTRY` — Docker image registry prefix (e.g. `ghcr.io/kokhp` or `docker.io/kokhp`)
- `IMAGE_TAG` — defaults to `latest`

## Deploying

Each container has a self-contained `deploy.sh`:

```bash
cd tts-clone/
./deploy.sh                 # builds, pushes, creates/updates serverless endpoint
```

See `tts-clone/deploy.sh` for the canonical flow. Other containers mirror it.

## Model storage

Weights persist across cold starts on a shared RunPod network volume mounted at `/runpod-volume`. Each container snapshots its own `models/<repo>/` subtree so cold starts after the first invocation only cost GPU warm-up (~15-45 s), not re-download.
