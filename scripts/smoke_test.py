#!/usr/bin/env python3
"""Gate 1: smoke-test each endpoint with a minimal /runsync call.

Reads endpoint IDs from Supabase tsage_api.runpod_endpoints and fires a
container-appropriate action. Records latency + output size + any error
back into the smoke_test_result JSONB column.

Cold-start on scale-to-zero endpoints with heavy model downloads can be
2-5 minutes on the first call. We allow up to 300s wall-clock per container
before marking it failed-timeout.
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

# Minimal action payloads per container. These exercise the smallest possible
# code path so cold-start models can be lazy-loaded on first real request.
# If a handler doesn't support "ping" it will error back fast, which is also
# useful smoke-signal.
PAYLOADS: dict[str, dict] = {
    "tts-clone":       {"input": {"action": "ping"}},
    "tts-fast":        {"input": {"action": "ping"}},
    "tts-design":      {"input": {"action": "ping"}},
    "tts-parler":      {"input": {"action": "ping"}},
    "tts-cosyvoice3":  {"input": {"action": "ping"}},
    "tts-cpu":         {"input": {"action": "ping"}},
    "asr-whisper":     {"input": {"action": "ping"}},
    "asr-omnilingual": {"input": {"action": "ping"}},
    "asr-realtime":    {"input": {"action": "ping"}},
    "asr-align":       {"input": {"action": "ping"}},
    "separator-gpu":   {"input": {"action": "ping"}},
    "separator-cpu":   {"input": {"action": "ping"}},
    "sfx":             {"input": {"action": "ping"}},
    "dialogue-en":     {"input": {"action": "ping"}},
    "vc":              {"input": {"action": "ping"}},
    "mt":              {"input": {"action": "ping"}},
    "agents-llm":      {"input": {"action": "ping"}},
    "agents-llm-big":  {"input": {"action": "ping"}},
    "lora-train":      {"input": {"action": "ping"}},
}

MAX_WAIT_S = 300


def _psql_select() -> list[tuple[str, str]]:
    """Return [(name, endpoint_id), ...] from Supabase."""
    out = subprocess.run(
        [
            "/opt/homebrew/opt/libpq/bin/psql", SUPABASE_DB_URL,
            "-At", "-F", "|",
            "-c", "SELECT name, endpoint_id FROM tsage_api.runpod_endpoints ORDER BY name;",
        ],
        check=True, capture_output=True, text=True,
    )
    rows = []
    for line in out.stdout.strip().splitlines():
        if not line.strip():
            continue
        name, eid = line.split("|", 1)
        rows.append((name, eid))
    return rows


def _smoke(endpoint_id: str, payload: dict) -> dict:
    """POST /runsync and return a result dict."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=MAX_WAIT_S) as r:
            raw = r.read().decode()
        dt = (time.time() - t0) * 1000
        parsed = json.loads(raw) if raw else {}
        return {
            "ok": True,
            "latency_ms": round(dt, 1),
            "output_bytes": len(raw),
            "status": parsed.get("status"),
            "preview": raw[:400],
        }
    except urllib.error.HTTPError as e:
        dt = (time.time() - t0) * 1000
        err = e.read().decode(errors="replace")[:800]
        return {
            "ok": False,
            "latency_ms": round(dt, 1),
            "http_code": e.code,
            "error": err,
        }
    except Exception as e:
        dt = (time.time() - t0) * 1000
        return {
            "ok": False,
            "latency_ms": round(dt, 1),
            "error": str(e),
        }


def _record(name: str, result: dict) -> None:
    """Write smoke-test result JSON into the row."""
    payload = json.dumps(result).replace("'", "''")
    subprocess.run(
        [
            "/opt/homebrew/opt/libpq/bin/psql", SUPABASE_DB_URL,
            "-v", "ON_ERROR_STOP=1",
            "-c",
            f"UPDATE tsage_api.runpod_endpoints "
            f"SET smoke_test_result = '{payload}'::jsonb, updated_at = NOW() "
            f"WHERE name = '{name}';",
        ],
        check=True, capture_output=True, text=True,
    )


def main() -> int:
    rows = _psql_select()
    print(f"Smoke-testing {len(rows)} endpoints...")
    good = []
    bad = []
    for name, eid in rows:
        key = name.replace("tsage-", "", 1)
        payload = PAYLOADS.get(key, {"input": {"action": "ping"}})
        print(f"\n--- {name} ({eid}) ---")
        result = _smoke(eid, payload)
        _record(name, result)
        if result.get("ok"):
            print(f"  OK  {result['latency_ms']} ms, {result['output_bytes']} B, status={result.get('status')}")
            good.append(name)
        else:
            print(f"  FAIL  {result.get('error', '')[:200]}")
            bad.append(name)

    print("\n\n=== SMOKE SUMMARY ===")
    print(f"ok:   {len(good)} / {len(rows)}")
    for n in good:
        print(f"  ok   {n}")
    for n in bad:
        print(f"  fail {n}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
