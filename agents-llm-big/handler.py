"""agents-llm-big — Premium LLM brain for complex Agents calls, backed by Qwen2.5-32B-Instruct int8.

Hardware: A100 80GB, ~35 GB VRAM at int8, scale-to-zero burst (user-triggered or auto-escalated from agents-llm).

Actions:
  chat(messages, tools=None, max_tokens=1024, temperature=0.7, stream=False) → assistant reply
"""
import os
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"

_engine = None


def _load():
    global _engine
    if _engine is not None:
        return _engine
    from huggingface_hub import snapshot_download
    local = snapshot_download(MODEL_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        _engine = {
            "engine": LLM(model=local, quantization="int8", gpu_memory_utilization=0.85, max_model_len=32768),
            "SamplingParams": SamplingParams,
        }
    except Exception:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(local)
        model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="auto")
        _engine = {"engine": "hf", "tok": tok, "model": model}
    return _engine


def action_chat(payload):
    messages = payload["messages"]
    tools = payload.get("tools")
    max_tokens = int(payload.get("max_tokens", 1024))
    temperature = float(payload.get("temperature", 0.7))

    engine = _load()
    if engine["engine"] == "hf":
        import torch
        tok = engine["tok"]
        model = engine["model"]
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"reply": reply, "model": MODEL_REPO}
    else:
        SamplingParams = engine["SamplingParams"]
        tok = engine["engine"].get_tokenizer()
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        result = engine["engine"].generate(prompt, sp)
        reply = result[0].outputs[0].text
        return {"reply": reply, "model": MODEL_REPO}


ACTIONS = {"chat": action_chat}


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
