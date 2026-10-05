# agents-llm-big

Premium LLM brain for complex Agents calls, backed by Qwen2.5-32B-Instruct int8. Scale-to-zero burst (user-triggered via `model=qwen2.5-32b` or auto-escalated from `agents-llm` when context/complexity demands).

## Hardware

- GPU: A100 80GB
- Steady VRAM: ~35 GB at int8
- Scale mode: scale-to-zero (burst)
- Cold start: ~25 s warm from volume cache; first request after idle ~45 s

## Model

| Repo | License | Context | Tool-call |
|------|---------|---------|-----------|
| `Qwen/Qwen2.5-32B-Instruct` | Apache 2.0 | 131K | native |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `chat` | `messages, tools=None, max_tokens=1024, temperature=0.7` | `reply, model, tool_calls?` |

## Environment variables

- `HF_TOKEN` — HuggingFace token

## Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "chat", "messages": [{"role":"user","content":"Explain Fermi paradox in 2 sentences."}], "max_tokens": 128}}'
```

## Deploy

```bash
export REGISTRY=ghcr.io/kokhp
./deploy.sh
```
