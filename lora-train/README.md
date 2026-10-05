# lora-train

Professional Voice Clone (PVC) LoRA fine-tune pipeline on CosyVoice2 base. User uploads 30+ min reference audio, we train a per-voice LoRA adapter for superior cloning fidelity beyond the zero-shot IVC baseline.

## Hardware

- GPU: L40S 48GB
- Peak VRAM: ~24 GB during training (base + LoRA + optimizer state + activations)
- Scale mode: scale-to-zero, user-triggered
- Training time: ~1 hour per voice (30-min reference input, 500 steps, bf16)

## Model

| Repo | License | Role |
|------|---------|------|
| `FunAudioLLM/CosyVoice2-0.5B` | Apache 2.0 | base — frozen during LoRA training |

## LoRA recipe

- Rank 32, alpha 64
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj` on the LM module
- bf16, AdamW, cosine LR 2e-4 → 2e-5
- 500 steps on 30-min reference audio with ~5× audio augmentation (pitch ±2 semitones, speed 0.9-1.1, light noise)
- Adapter saved to `/runpod-volume/pvc-adapters/{voice_id}/`

## Actions

| Action | Input | Output |
|--------|-------|--------|
| `train` | `voice_id?, ref_audio_url, user_id, hours_target?=0.5` | `job_id, voice_id, status` |
| `status` | `job_id` | `{status: queued\|preparing\|training\|complete\|failed, progress, adapter_url?, error?}` |

## Flow

1. User posts to `/v1/voices/{id}/professional` on orchestrator (billed $29 one-time).
2. Orchestrator fires `train` action here, gets `job_id`.
3. Client polls `status(job_id)` every 30s.
4. On `complete`, adapter is at `/runpod-volume/pvc-adapters/{voice_id}/` — accessible from `tts-clone` worker (shared volume).
5. Subsequent `tts-clone.synthesize(voice_id=X)` loads the adapter and runs CosyVoice2 with PVC fidelity.

## Environment variables

- `HF_TOKEN` — HuggingFace token

## Smoke test

```bash
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "train", "ref_audio_url": "https://.../30min.wav", "user_id": "u_test"}}'
```

Note: this scaffold uses a simplified placeholder for the actual CosyVoice fine-tune step. Production implementation requires wiring CosyVoice's native trainer entrypoint; the handler + job tracking shell is production-ready.

## Deploy

```bash
export REGISTRY=ghcr.io/kokhp
./deploy.sh
```
