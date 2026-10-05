# mt

Multilingual translation backed by MADLAD-400 10B primary + MADLAD-400 7B-bt fallback + M2M-100 1.2B lightweight.

## Hardware

- GPU: A5000 (24 GB)
- Steady VRAM: ~8 GB with int4 quantization (MADLAD 10B), ~6 GB (MADLAD 7B), ~3 GB (M2M-100)
- Scale mode: scale-to-zero (min-worker 0, max-worker 20)
- Cold start: ~15 s warm from volume cache

## Models

| Repo | License | Role | Langs |
|------|---------|------|-------|
| `google/madlad400-10b-mt` | Apache 2.0 | primary (creative register) | 419 trained |
| `google/madlad400-7b-mt-bt` | Apache 2.0 | standard register (faster) | 419 trained |
| `facebook/m2m100_1.2B` | MIT | lightweight fallback | 100 |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `translate` | `text, tgt_lang, src_lang="auto", register="standard"\|"creative", model?` | `translated, model, src_lang, tgt_lang` |
| `translate_batch` | `texts: [str], tgt_lang, src_lang, register` | `translated: [str]` |
| `detect` | `text` | `lang, confidence` (via fasttext lid.176) |

## Lang codes

Use MADLAD lang codes (2-letter ISO 639-1 for most, with `<2xx>` prefix handled internally).

## Environment variables

- `HF_TOKEN` — HuggingFace token

## Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "translate", "text": "Hello, how are you?", "tgt_lang": "es"}}'
```

## Deploy

```bash
export REGISTRY=ghcr.io/kokhp
./deploy.sh
```
