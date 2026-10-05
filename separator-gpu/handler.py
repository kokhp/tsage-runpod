"""separator-gpu — BS-RoFormer vocal / music-element separation.

Hardware: RTX 4090 (24 GB), ~8 GB VRAM, scale-to-zero.

Actions:
  separate(audio_url, mode="vocals"|"me"|"both", stem="vocals"|"instrumental")
    → {vocals_b64, instrumental_b64, me_b64, sample_rate}
"""
import os
import io
import base64
import tempfile
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

BS_REPO = "lucidrains/BS-RoFormer"  # weights come from ZFTurbo zoo mirrored on HF

_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    import torch
    from bs_roformer import BSRoformer  # type: ignore
    ckpt_path = Path(WEIGHTS_DIR) / "bs-roformer" / "model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        import requests
        url = "https://huggingface.co/KimberleyJensen/BS-Roformer/resolve/main/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(ckpt_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    model = BSRoformer(
        dim=512, depth=12, stereo=True, num_stems=1, time_transformer_depth=1,
        freq_transformer_depth=1,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.cuda().eval()
    _model = model
    return _model


def _fetch_file(url_or_b64, suffix=".wav"):
    import requests
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if url_or_b64.startswith("http"):
        tmp.write(requests.get(url_or_b64, timeout=120).content)
    else:
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio.T if audio.ndim == 2 else audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_separate(payload):
    import torch, numpy as np, soundfile as sf
    model = _load()
    mode = payload.get("mode", "vocals")
    audio_fp = _fetch_file(payload["audio_url"])
    try:
        audio, sr = sf.read(audio_fp, dtype="float32")
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        else:
            audio = audio.T  # (channels, samples)
        inp = torch.from_numpy(audio).unsqueeze(0).cuda()
        with torch.no_grad():
            vocals = model(inp).squeeze(0).cpu().numpy()
        instrumental = audio - vocals
        out = {"sample_rate": sr}
        if mode in ("vocals", "both"):
            out["vocals_b64"] = _pack_wav(vocals, sr)
        if mode in ("me", "both", "instrumental"):
            out["instrumental_b64"] = _pack_wav(instrumental, sr)
        return out
    finally:
        try: os.unlink(audio_fp)
        except: pass


ACTIONS = {"separate": action_separate}


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
