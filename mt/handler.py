"""mt — Multilingual translation API backed by MADLAD-400 10B (GPTQ-int4)
with MADLAD-400 7B-bt + M2M-100 1.2B as fallback.

Hardware: A5000 (24 GB), ~8 GB VRAM with int4 quantization, scale-to-zero.

Actions:
  translate(text, src_lang="auto", tgt_lang, register="standard") → translated_text
  translate_batch(texts, src_lang, tgt_lang, register)            → [translated_texts]
  detect(text)                                                     → detected_lang (via MADLAD)
"""
import os
import time
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

PRIMARY_REPO = "google/madlad400-10b-mt"
FALLBACK_REPO = "google/madlad400-7b-mt-bt"
LIGHTWEIGHT_REPO = "facebook/m2m100_1.2B"

_primary = None  # (tokenizer, model) tuple
_fallback = None
_lightweight = None


def _load_primary():
    """MADLAD-400 10B. Loaded in bf16 or int4 depending on availability."""
    global _primary
    if _primary is not None:
        return _primary
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    import torch

    local = snapshot_download(PRIMARY_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    tok = T5Tokenizer.from_pretrained(local)
    # Try int4 via bitsandbytes first for memory savings, else bf16
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = T5ForConditionalGeneration.from_pretrained(local, quantization_config=bnb, device_map="auto")
    except Exception:
        model = T5ForConditionalGeneration.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="auto")
    _primary = (tok, model)
    return _primary


def _load_fallback():
    global _fallback
    if _fallback is not None:
        return _fallback
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    from huggingface_hub import snapshot_download
    import torch

    local = snapshot_download(FALLBACK_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    tok = T5Tokenizer.from_pretrained(local)
    model = T5ForConditionalGeneration.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="auto")
    _fallback = (tok, model)
    return _fallback


def _load_lightweight():
    global _lightweight
    if _lightweight is not None:
        return _lightweight
    from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
    from huggingface_hub import snapshot_download

    local = snapshot_download(LIGHTWEIGHT_REPO, cache_dir=WEIGHTS_DIR, token=os.environ.get("HF_TOKEN"))
    tok = M2M100Tokenizer.from_pretrained(local)
    model = M2M100ForConditionalGeneration.from_pretrained(local, device_map="auto")
    _lightweight = (tok, model)
    return _lightweight


def _madlad_prompt(text, tgt_lang):
    """MADLAD uses a `<2xx>` style target-lang prefix."""
    return f"<2{tgt_lang}> {text}"


def action_translate(payload):
    text = payload["text"]
    tgt_lang = payload["tgt_lang"]
    src_lang = payload.get("src_lang", "auto")
    register = payload.get("register", "standard")  # "creative" uses larger model
    model_pref = payload.get("model", "auto")  # "madlad_10b" | "madlad_7b" | "m2m_1.2b"

    if model_pref == "m2m_1.2b":
        tok, model = _load_lightweight()
        tok.src_lang = src_lang if src_lang != "auto" else "en"
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, forced_bos_token_id=tok.get_lang_id(tgt_lang), max_new_tokens=512)
        return {"translated": tok.batch_decode(out, skip_special_tokens=True)[0], "model": "m2m100_1.2b"}

    try:
        if register == "creative" or model_pref == "madlad_10b":
            tok, model = _load_primary()
            model_name = "madlad400-10b-mt"
        else:
            tok, model = _load_fallback()
            model_name = "madlad400-7b-mt-bt"
    except Exception:
        tok, model = _load_fallback()
        model_name = "madlad400-7b-mt-bt-fallback"

    prompt = _madlad_prompt(text, tgt_lang)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(**enc, max_new_tokens=1024, num_beams=4, early_stopping=True)
    translated = tok.batch_decode(out, skip_special_tokens=True)[0]
    return {"translated": translated, "model": model_name, "src_lang": src_lang, "tgt_lang": tgt_lang}


def action_translate_batch(payload):
    texts = payload["texts"]
    tgt_lang = payload["tgt_lang"]
    src_lang = payload.get("src_lang", "auto")
    register = payload.get("register", "standard")
    out = []
    for t in texts:
        r = action_translate({"text": t, "tgt_lang": tgt_lang, "src_lang": src_lang, "register": register})
        out.append(r.get("translated"))
    return {"translated": out, "src_lang": src_lang, "tgt_lang": tgt_lang}


def action_detect(payload):
    """MADLAD doesn't expose language detection directly; use a simple classifier via fasttext if available."""
    text = payload["text"]
    try:
        import fasttext
        model_path = Path(WEIGHTS_DIR) / "lid.176.ftz"
        if not model_path.exists():
            import urllib.request
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz", str(model_path))
        ft = fasttext.load_model(str(model_path))
        labels, probs = ft.predict(text.replace("\n", " "), k=1)
        lang = labels[0].replace("__label__", "")
        return {"lang": lang, "confidence": float(probs[0])}
    except Exception as e:
        return {"error": f"lang detect unavailable: {e}"}


ACTIONS = {
    "translate": action_translate,
    "translate_batch": action_translate_batch,
    "detect": action_detect,
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
