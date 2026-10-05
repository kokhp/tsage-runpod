#!/usr/bin/env python3
"""Gate 1 re-smoke for the 5 containers fixed in commit ea97ee6.

- Fires a container-appropriate /runsync against RunPod.
- Treats HTTP 503 (backend_warming) as "cold, retry" up to 5 minutes.
- Upserts the final JSON result into tsage_api.runpod_endpoints.smoke_test_result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

RUNPOD_API_KEY  = os.environ["RUNPOD_API_KEY"]
SUPABASE_DB_URL = os.environ["SUPABASE_DB_URL"]
PSQL            = "/opt/homebrew/opt/libpq/bin/psql"

ENDPOINTS = {
    "tts-cosyvoice3": "8spc9ovaqk5p98",
    "tts-cpu":        "k0amgkplyk5gci",
    "asr-whisper":    "xr6sy36x0jd8pi",
    "asr-realtime":   "y1inm3w9244j0q",
    "sfx":            "uaeu32866j8b4y",
}

# Smallest useful payload per handler.
PAYLOADS: dict[str, dict] = {
    "tts-cosyvoice3": {"input": {"action": "synthesize", "text": "Hello world", "voice_id": "default"}},
    "tts-cpu":        {"input": {"action": "list_voices"}},
    "asr-whisper":    {"input": {"action": "transcribe",
                                  "audio_url": "https://filesamples.com/samples/audio/wav/sample3.wav",
                                  "model": "turbo"}},
    "asr-realtime":   {"input": {"action": "transcribe",
                                  "audio_url": "https://filesamples.com/samples/audio/wav/sample3.wav"}},
    "sfx":            {"input": {"action": "generate", "prompt": "soft rain", "duration_s": 3}},
}

RETRY_BUDGET_S = 300          # 5-min overall budget per endpoint
BETWEEN_RETRIES_S = 20


def _runsync_once(endpoint_id: str, payload: dict, timeout_s: int) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode()
        dt = (time.time() - t0) * 1000
        parsed = json.loads(raw) if raw else {}
        return {"ok": True, "latency_ms": round(dt, 1),
                "status": parsed.get("status"),
                "runpod_status": parsed.get("status"),
                "output_bytes": len(raw),
                "preview": raw[:500],
                "parsed": parsed}
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")[:600]
        return {"ok": False, "http_code": e.code, "error": err,
                "latency_ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1)}


def smoke(name: str, endpoint_id: str) -> dict:
    payload = PAYLOADS[name]
    deadline = time.time() + RETRY_BUDGET_S
    attempts = 0
    last = {}
    while time.time() < deadline:
        attempts += 1
        remaining = max(30, int(deadline - time.time()))
        last = _runsync_once(endpoint_id, payload, remaining)
        # Done if we have a response AND it isn't a transient warmup error.
        if last.get("ok"):
            # RunPod serverless returns status COMPLETED | FAILED | IN_QUEUE | IN_PROGRESS
            # /runsync blocks until COMPLETED/FAILED unless timeout fires.
            parsed = last.get("parsed", {})
            runpod_status = parsed.get("status")
            output = parsed.get("output", {})
            # Treat as a real pass only if handler itself did not error.
            if isinstance(output, dict) and output.get("error"):
                last["ok"] = False
                last["reason"] = f"handler-error: {str(output.get('error'))[:300]}"
                last["handler_output"] = output
                return _finalize(last, attempts)
            if runpod_status in ("COMPLETED",):
                return _finalize(last, attempts)
            if runpod_status in ("FAILED",):
                last["ok"] = False
                last["reason"] = "runpod status=FAILED"
                return _finalize(last, attempts)
            # Some other state (IN_QUEUE / IN_PROGRESS) — unexpected on /runsync
            time.sleep(BETWEEN_RETRIES_S)
            continue
        code = last.get("http_code")
        if code == 503 or "backend_warming" in str(last.get("error", "")).lower():
            # Cold start — retry.
            print(f"    [{name}] cold ({code or ''}), retrying in {BETWEEN_RETRIES_S}s "
                  f"(attempt {attempts})")
            time.sleep(BETWEEN_RETRIES_S)
            continue
        # Other error — don't thrash.
        return _finalize(last, attempts)
    last["reason"] = f"budget exhausted after {attempts} attempts"
    return _finalize(last, attempts)


def _finalize(result: dict, attempts: int) -> dict:
    result["attempts"] = attempts
    result["ts"] = int(time.time())
    # Drop giant parsed body from stored JSON — keep status + any error + preview.
    parsed = result.pop("parsed", None)
    if parsed is not None:
        result["output_summary"] = _summarize_output(parsed.get("output"))
    return result


def _summarize_output(output):
    if output is None:
        return None
    if isinstance(output, dict):
        summary = {}
        for k, v in output.items():
            if isinstance(v, str) and len(v) > 160:
                summary[k] = f"<{len(v)} chars>"
            elif isinstance(v, (list, tuple)) and len(v) > 20:
                summary[k] = f"<list len={len(v)}>"
            else:
                summary[k] = v
        return summary
    if isinstance(output, str):
        return output[:200]
    return str(output)[:200]


def upsert(name: str, result: dict) -> None:
    payload = json.dumps(result).replace("'", "''")
    subprocess.run(
        [PSQL, SUPABASE_DB_URL, "-v", "ON_ERROR_STOP=1", "-c",
         f"UPDATE tsage_api.runpod_endpoints "
         f"SET smoke_test_result = '{payload}'::jsonb, updated_at = NOW() "
         f"WHERE name = '{name}';"],
        check=True, capture_output=True, text=True,
    )


def main() -> int:
    only = sys.argv[1:]
    results = {}
    for name, eid in ENDPOINTS.items():
        if only and name not in only:
            continue
        print(f"\n=== {name} ({eid}) ===")
        res = smoke(name, eid)
        upsert(name, res)
        results[name] = res
        tag = "OK" if res.get("ok") else "FAIL"
        print(f"  {tag}  attempts={res.get('attempts')}  latency={res.get('latency_ms')} ms")
        if not res.get("ok"):
            print(f"  reason: {res.get('reason') or res.get('error', '')[:300]}")
        else:
            print(f"  output: {res.get('output_summary')}")
    print("\n--- SUMMARY ---")
    for n, r in results.items():
        print(f"  {'ok  ' if r.get('ok') else 'FAIL'} {n}")
    return 0 if all(r.get("ok") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
