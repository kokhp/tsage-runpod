"""sfx — Riffusion v1 SFX / foley generation.

Hardware: A4000 (16 GB), ~5 GB VRAM, scale-to-zero.

Actions:
  generate(prompt, duration_s=5, seed?, format="wav") → audio_b64, spectrogram_png_b64
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

RIFF_REPO = "riffusion/riffusion-model-v1"

_pipe = None


def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionPipeline  # type: ignore
    from huggingface_hub import snapshot_download
    local = snapshot_download(RIFF_REPO, cache_dir=WEIGHTS_DIR,
                              token=os.environ.get("HF_TOKEN"))
    _pipe = StableDiffusionPipeline.from_pretrained(local, torch_dtype=torch.float16,
                                                     safety_checker=None).to("cuda")
    try:
        _pipe.enable_attention_slicing()
    except Exception:
        pass
    return _pipe


def _spectrogram_to_wav(image, duration_s=5):
    """Riffusion v1 trick: PIL spectrogram → audio via Griffin-Lim."""
    import numpy as np, librosa
    arr = np.array(image.convert("L"), dtype=np.float32)
    # Riffusion normalises to [0, 255]; map back to log-magnitude
    mag = (arr / 255.0) * 10.0  # heuristic normalisation
    mag = np.flipud(mag)
    linear = librosa.db_to_amplitude(mag * 10 - 60)
    audio = librosa.griffinlim(linear, hop_length=256, n_iter=32)
    sr = 22050
    # Trim/pad to requested duration
    target = int(duration_s * sr)
    if len(audio) >= target:
        audio = audio[:target]
    else:
        pad = np.zeros(target - len(audio), dtype=audio.dtype)
        audio = np.concatenate([audio, pad])
    return audio, sr


def action_generate(payload):
    import torch, io as _io, soundfile as sf
    pipe = _load()
    prompt = payload["prompt"]
    duration_s = int(payload.get("duration_s", 5))
    seed = payload.get("seed")
    gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

    image = pipe(prompt, width=512, height=512, num_inference_steps=50,
                 generator=gen).images[0]

    audio, sr = _spectrogram_to_wav(image, duration_s=duration_s)

    wav_buf = _io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")

    png_buf = _io.BytesIO()
    image.save(png_buf, format="PNG")

    return {
        "audio_b64": base64.b64encode(wav_buf.getvalue()).decode(),
        "spectrogram_png_b64": base64.b64encode(png_buf.getvalue()).decode(),
        "sample_rate": sr,
        "duration_s": duration_s,
        "prompt": prompt,
    }


ACTIONS = {"generate": action_generate}


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
