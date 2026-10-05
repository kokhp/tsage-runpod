# agents-llm

LLM brain for Agents / Reception AI. Qwen3-14B int8 primary with Mistral-Nemo-12B fallback.

## Hardware

- GPU: A5000 (24 GB)
- Steady VRAM: ~15 GB at int8 (Qwen3-14B), ~12 GB (Mistral-Nemo-12B)
- Scale mode: **min-worker 1** (streaming/agents cannot cold-start)
- Cold start: ~15 s warm from volume cache

## Models

| Repo | License | Role | Context | Tool-call |
|------|---------|------|---------|-----------|
| `Qwen/Qwen3-14B` | Apache 2.0 | primary | 32K / 131K YaRN | native + MCP |
| `mistralai/Mistral-Nemo-Instruct-2407` | Apache 2.0 | fallback | 128K | native |

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `chat` | `messages, tools=None, max_tokens=512, temperature=0.7, model?` | `reply, model, tool_calls?` |
| `chat_stream` | same as chat | token stream (via RunPod async /run endpoint) |

## Function calling / tools

Pass OpenAI-style `tools` schema. Qwen3-14B emits structured tool calls via its native chat template; the handler surfaces them as `tool_calls` in the response. MCP integration via Qwen-Agent (external orchestration).

## Environment variables

- `HF_TOKEN` — HuggingFace token

## Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "chat", "messages": [{"role":"user","content":"Hi"}], "max_tokens": 32}}'
```

## Deploy

```bash
export REGISTRY=ghcr.io/kokhp
./deploy.sh
```
