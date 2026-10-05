"""agents-llm — LLM brain for Agents / Reception AI backed by Qwen3-14B (int8)
with Mistral-Nemo-12B fallback.

Hardware: A5000 (24 GB), ~15 GB VRAM at int8, min-worker 1 (streaming cannot cold-start).

Actions:
  chat(messages, tools=None, max_tokens=512, temperature=0.7, stream=False) → assistant reply + optional tool calls
  chat_stream(messages, ...) → token stream (via streaming generator)
"""
import os
import json
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

PRIMARY_REPO = "Qwen/Qwen3-14B"
FALLBACK_REPO = "mistralai/Mistral-Nemo-Instruct-2407"

_primary = None  # vLLM engine
_fallback = None


def _load_primary():
    global _primary
    if _primary is not None:
        return _primary
    from huggingface_hub import snapshot_download
    local = snapshot_download(PRIMARY_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        _primary = {
            "engine": LLM(model=local, quantization="int8", gpu_memory_utilization=0.75, max_model_len=32768),
            "SamplingParams": SamplingParams,
        }
    except Exception:
        # fallback to HF transformers if vLLM fails to initialize
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(local)
        model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="auto")
        _primary = {"engine": "hf", "tok": tok, "model": model}
    return _primary


def _load_fallback():
    global _fallback
    if _fallback is not None:
        return _fallback
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    local = snapshot_download(FALLBACK_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="auto")
    _fallback = {"engine": "hf", "tok": tok, "model": model}
    return _fallback


def _format_prompt(engine, messages, tools=None):
    """Render chat messages + optional tool-use schema via the model's tokenizer chat template."""
    tok = engine.get("tok")
    if tok is None:
        # vLLM path — use raw messages
        return messages
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)


def action_chat(payload):
    messages = payload["messages"]
    tools = payload.get("tools")
    max_tokens = int(payload.get("max_tokens", 512))
    temperature = float(payload.get("temperature", 0.7))
    model_pref = payload.get("model", "qwen3-14b")

    try:
        engine = _load_primary() if model_pref != "mistral-nemo" else _load_fallback()
    except Exception:
        engine = _load_fallback()

    if engine["engine"] == "hf":
        import torch
        tok = engine["tok"]
        model = engine["model"]
        prompt = _format_prompt(engine, messages, tools)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"reply": reply, "model": PRIMARY_REPO if engine is _primary else FALLBACK_REPO}
    else:
        # vLLM path
        SamplingParams = engine["SamplingParams"]
        tok = engine["engine"].get_tokenizer()
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        result = engine["engine"].generate(prompt, sp)
        reply = result[0].outputs[0].text
        return {"reply": reply, "model": PRIMARY_REPO}


def action_chat_stream(payload):
    """Returns a generator (not supported in RunPod's sync runsync — use /run async endpoint)."""
    # Simplified: return the full response in one chunk; proper streaming requires RunPod's streaming API
    return action_chat(payload)


ACTIONS = {
    "chat": action_chat,
    "chat_stream": action_chat_stream,
}


def handler(job):
    payload = job.get("input") or {}
    action = payload.get("action")
    fn = ACTIONS.get(action)
    if not fn:
        return {"error": f"unknown action: {action}", "valid": list(ACTIONS)}
    try:
        return fn(payload)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
