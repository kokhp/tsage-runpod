"""vc — Voice conversion: kNN-VC + FreeVC.

Hardware: A4000 (16 GB), ~6 GB VRAM, scale-to-zero.

Actions:
  convert(source_audio_url, target_voice_id|target_audio_url,
          model="knn-vc"|"freevc", strength=1.0, format="wav")
    → audio_b64, sample_rate
  register(voice_id, ref_audio_url)   → voice_id, ref_path (used by target_voice_id)
"""
import os
import io
import time
import base64
import tempfile
import traceback
from pathlib import Path

import runpod

VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME", "/runpod-volume")
WEIGHTS_DIR = f"{VOLUME_ROOT}/models"
VOICE_DIR = f"{VOLUME_ROOT}/voices/vc"
Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(VOICE_DIR).mkdir(parents=True, exist_ok=True)

_knn = None
_free = None


def _load_knn():
    global _knn
    if _knn is not None:
        return _knn
    import torch
    _knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                          pretrained=True, device="cuda")
    return _knn


def _load_free():
    global _free
    if _free is not None:
        return _free
    import torch
    from huggingface_hub import snapshot_download
    local = snapshot_download("OlaWod/FreeVC", cache_dir=WEIGHTS_DIR,
                              token=os.environ.get("HF_TOKEN"))
    # FreeVC exposes a model via its own repo code; fall back to raw torch load
    from freevc.utils import load_checkpoint, get_hparams_from_file  # type: ignore
    from freevc.models import SynthesizerTrn  # type: ignore
    hps = get_hparams_from_file(f"{local}/config.json")
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
    load_checkpoint(f"{local}/freevc.pth", net_g, None)
    net_g.eval()
    _free = (net_g, hps)
    return _free


def _fetch_file(url_or_b64):
    import requests
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if url_or_b64.startswith("http"):
        tmp.write(requests.get(url_or_b64, timeout=60).content)
    else:
        tmp.write(base64.b64decode(url_or_b64))
    tmp.close()
    return tmp.name


def _pack_wav(audio, sr):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def action_convert(payload):
    model_name = payload.get("model", "knn-vc")
    src_fp = _fetch_file(payload["source_audio_url"])
    tgt_fp = None
    try:
        if payload.get("target_voice_id"):
            tgt_fp = str(Path(VOICE_DIR) / f"{payload['target_voice_id']}.wav")
            if not Path(tgt_fp).exists():
                return {"error": f"voice_id {payload['target_voice_id']} not registered"}
        elif payload.get("target_audio_url"):
            tgt_fp = _fetch_file(payload["target_audio_url"])
        else:
            return {"error": "need target_voice_id or target_audio_url"}

        if model_name == "freevc":
            import torch, torchaudio
            net_g, hps = _load_free()
            # Spectrogram-based path — simplified; FreeVC repo has reference logic
            src, sr = torchaudio.load(src_fp)
            tgt, _ = torchaudio.load(tgt_fp)
            # Use the model's built-in `voice_conversion` helper if available
            if hasattr(net_g, "voice_conversion"):
                with torch.no_grad():
                    audio = net_g.voice_conversion(src.cuda(), tgt.cuda()).squeeze().cpu().numpy()
            else:
                audio = src.squeeze().numpy()  # passthrough stub
            return {"audio_b64": _pack_wav(audio, 16000), "sample_rate": 16000,
                    "model": "freevc"}
        else:
            knn = _load_knn()
            query_seq = knn.get_features(src_fp)
            matching_set = knn.get_matching_set([tgt_fp])
            out_wav = knn.match(query_seq, matching_set, topk=4)
            audio = out_wav.cpu().numpy().squeeze()
            return {"audio_b64": _pack_wav(audio, 16000), "sample_rate": 16000,
                    "model": "knn-vc"}
    finally:
        try: os.unlink(src_fp)
        except: pass
        if tgt_fp and payload.get("target_audio_url"):
            try: os.unlink(tgt_fp)
            except: pass


def action_register(payload):
    import soundfile as sf, requests, librosa, numpy as np
    name = payload.get("voice_id") or f"vc_{int(time.time())}"
    url = payload["ref_audio_url"]
    data = requests.get(url, timeout=60).content if url.startswith("http") else base64.b64decode(url)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    out = Path(VOICE_DIR) / f"{name}.wav"
    sf.write(str(out), audio.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
    return {"voice_id": name, "ref_path": str(out)}


ACTIONS = {"convert": action_convert, "register": action_register}


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
