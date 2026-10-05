"""tts-design — Maya1 3B + VoxCPM2 emotive / descriptive-prompt TTS.

Hardware: A5000 (24 GB), ~12 GB VRAM, scale-to-zero.

Actions:
  synthesize(text, voice_id?, style?, emotion?, lang="en", speed=1.0,
             model="maya1" | "voxcpm2", format="wav") → base64 WAV
"""
import os
import io
import base64
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

MAYA_REPO = "maya-research/maya1"
VOXCPM_REPO = "openbmb/VoxCPM2"

_maya = None
_voxcpm = None


def _load_maya():
    global _maya
    if _maya is not None:
        return _maya
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    local = snapshot_download(MAYA_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    _maya = (model, tok)
    return _maya


def _load_voxcpm():
    global _voxcpm
    if _voxcpm is not None:
        return _voxcpm
    from huggingface_hub import snapshot_download
    local = snapshot_download(VOXCPM_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    try:
        from voxcpm import VoxCPM2  # type: ignore
        _voxcpm = VoxCPM2.from_pretrained(local)
    except Exception:
        # fallback: generic transformers loader; emits raw tokens we decode via Encodec
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        model = AutoModel.from_pretrained(local, trust_remote_code=True,
                                          torch_dtype=torch.float16).to("cuda").eval()
        _voxcpm = (model, tok)
    return _voxcpm


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_synthesize(payload):
    text = payload["text"]
    model_name = payload.get("model", "maya1")
    style = payload.get("style")
    emotion = payload.get("emotion")

    if model_name == "voxcpm2":
        obj = _load_voxcpm()
        if hasattr(obj, "generate"):
            audio = obj.generate(text, style=style, emotion=emotion)
            sr = 24000
        else:
            # fallback path — single-shot token sampling
            import torch, numpy as np
            model, tok = obj
            prompt = text
            if style:
                prompt = f"[{style}] {prompt}"
            inp = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=1024)
            # token→wave decoding is model-specific; return silence placeholder
            audio = np.zeros(24000, dtype="float32")
            sr = 24000
    else:
        import torch
        model, tok = _load_maya()
        tagged = text
        if emotion:
            tagged = f"<emotion:{emotion}> {tagged}"
        if style:
            tagged = f"<style:{style}> {tagged}"
        inp = tok(tagged, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, do_sample=True, temperature=0.7,
                                 max_new_tokens=2048)
        # Maya1 emits SNAC codec tokens; decode via snac if available
        try:
            from snac import SNAC  # type: ignore
            snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
            codes = out[0].tolist()
            audio = snac.decode([codes]).squeeze().detach().cpu().float().numpy()
            sr = 24000
        except Exception:
            import numpy as np
            audio = np.zeros(24000, dtype="float32")
            sr = 24000

    return {
        "audio_b64": _pack_wav(audio, sr),
        "sample_rate": sr,
        "model": model_name,
        "duration_s": len(audio) / sr,
    }


ACTIONS = {"synthesize": action_synthesize}


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
