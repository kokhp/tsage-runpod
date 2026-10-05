# tts-clone

Voice-clone TTS backed by CosyVoice2-0.5B with optional OpenVoice V2 tone-color conversion for extra accent control.

## Hardware

- GPU: A5000 (24 GB)
- Steady VRAM: ~6 GB
- Scale mode: scale-to-zero (min-worker 0, max-worker 10)
- Cold start: ~25 s after first warm pull (weights on `/runpod-volume`)

## Models

| Repo | Role | Pin |
|------|------|-----|
| `FunAudioLLM/CosyVoice2-0.5B` | zero-shot clone + base TTS | main (snapshot_download pins to current revision at build time) |
| `myshell-ai/OpenVoiceV2` | tone-color converter | main |

The containers call `huggingface_hub.snapshot_download(..., cache_dir=/runpod-volume/models)` on cold start. The first worker pulls weights (~2.1 GB CosyVoice + ~380 MB OpenVoice), subsequent workers reuse the volume cache.

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `synthesize` | `text, voice_id?, lang?="en", emotion?, speed?=1.0, format?="wav", stream?=false` | `audio_b64, sample_rate=22050, voice_id, lang, duration_s` |
| `clone` | `ref_audio_url, name?` | `voice_id, ref_path` |
| `list_voices` | - | `voices: [{voice_id, ref_path, created_at}]` |

## Environment variables

- `HF_TOKEN` — HuggingFace token for private/gated snapshot downloads
- `RUNPOD_VOLUME` — defaults to `/runpod-volume`; model + voice cache lives here

## Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "synthesize", "text": "hello world", "lang": "en"}}'
```

## Deploy

```bash
export REGISTRY=ghcr.io/kokhp
./deploy.sh
```
