#!/usr/bin/env python3
"""Gate 1: create 19 RunPod serverless endpoints via REST API.

Flow per container:
  1. POST /templates  -> templateId  (image + env vars)
  2. POST /endpoints  -> endpointId  (gpu/cpu + workers + templateId)
  3. INSERT into tsage_api.runpod_endpoints

min_workers=0 on all; operator will bump realtime endpoints manually.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN       = os.environ["HF_TOKEN"]
SUPABASE_DB_URL = os.environ["SUPABASE_DB_URL"]

REST = "https://rest.runpod.io/v1"


@dataclass
class Container:
    name: str
    gpu_tier: str
    gpu_ids: list[str] | None   # None for CPU
    cpu_flavors: list[str] | None  # None for GPU
    container_disk_gb: int = 50


# GPU IDs are RunPod's exact strings per /v1/openapi.json
GPU_A5000  = ["NVIDIA RTX A5000"]
GPU_A4000  = ["NVIDIA RTX A4000"]
GPU_4090   = ["NVIDIA GeForce RTX 4090"]
GPU_A100   = ["NVIDIA A100 80GB PCIe"]
GPU_L40S   = ["NVIDIA L40S"]
CPU_LARGE  = ["cpu3g", "cpu5g"]  # general-purpose CPU; falls back cpu3g→cpu5g

CONTAINERS: list[Container] = [
    # A5000 (24 GB)
    Container("tts-clone",       "A5000", GPU_A5000, None),
    Container("tts-design",      "A5000", GPU_A5000, None),
    Container("asr-whisper",     "A5000", GPU_A5000, None),
    Container("asr-omnilingual", "A5000", GPU_A5000, None),
    Container("mt",              "A5000", GPU_A5000, None),
    Container("agents-llm",      "A5000", GPU_A5000, None),
    # A4000 (16 GB)
    Container("tts-fast",        "A4000", GPU_A4000, None),
    Container("tts-parler",      "A4000", GPU_A4000, None),
    Container("tts-cosyvoice3",  "A4000", GPU_A4000, None),
    Container("asr-realtime",    "A4000", GPU_A4000, None),
    Container("asr-align",       "A4000", GPU_A4000, None),
    Container("sfx",             "A4000", GPU_A4000, None),
    Container("vc",              "A4000", GPU_A4000, None),
    # RTX 4090 (24 GB)
    Container("separator-gpu",   "RTX 4090", GPU_4090, None),
    Container("dialogue-en",     "RTX 4090", GPU_4090, None),
    # A100 80GB
    Container("agents-llm-big",  "A100 80GB", GPU_A100, None),
    # L40S (48 GB)
    Container("lora-train",      "L40S", GPU_L40S, None),
    # CPU-only
    Container("tts-cpu",         "CPU",  None, CPU_LARGE, container_disk_gb=20),
    Container("separator-cpu",   "CPU",  None, CPU_LARGE, container_disk_gb=20),
]


def _api(method: str, path: str, body: dict | None = None, retries: int = 3) -> dict:
    url = f"{REST}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            last_err = RuntimeError(f"HTTP {e.code} {method} {path}: {err_body}")
            # 429 and 5xx: retry
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise last_err
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_err  # unreachable


def _list_templates() -> list[dict]:
    return _api("GET", "/templates")


def _list_endpoints() -> list[dict]:
    return _api("GET", "/endpoints")


def _create_template(c: Container) -> str:
    name = f"tsage-{c.name}"
    image = f"ghcr.io/kokhp/tsage-runpod-{c.name}:latest"

    # Reuse existing template with same name if present
    for t in _list_templates():
        if t.get("name") == name:
            print(f"  reuse template {t['id']} for {name}")
            return t["id"]

    body = {
        "name": name,
        "imageName": image,
        "isServerless": True,
        "category": "CPU" if c.cpu_flavors else "NVIDIA",
        "containerDiskInGb": c.container_disk_gb,
        "volumeInGb": 0,  # scale-to-zero, no persistent volume
        "env": {
            "HF_TOKEN": HF_TOKEN,
            "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
            "RUNPOD_VOLUME": "/runpod-volume",
        },
        "readme": f"RunPod serverless template for tsage-{c.name}. Managed by scripts/create_endpoints.py.",
    }
    r = _api("POST", "/templates", body)
    tid = r["id"]
    print(f"  template {tid} created for {name}")
    return tid


def _create_endpoint(c: Container, template_id: str) -> str:
    name = f"tsage-{c.name}"

    # Reuse existing endpoint by name
    for e in _list_endpoints():
        if e.get("name") == name:
            print(f"  reuse endpoint {e['id']} for {name}")
            return e["id"]

    body: dict = {
        "name": name,
        "templateId": template_id,
        "workersMin": 0,
        "workersMax": 10,
        "idleTimeout": 5,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "executionTimeoutMs": 600_000,
        "flashboot": True,
    }
    if c.cpu_flavors:
        body["computeType"] = "CPU"
        body["cpuFlavorIds"] = c.cpu_flavors
        body["vcpuCount"]    = 4
    else:
        body["computeType"] = "GPU"
        body["gpuTypeIds"]  = c.gpu_ids
        body["gpuCount"]    = 1

    r = _api("POST", "/endpoints", body)
    eid = r["id"]
    print(f"  endpoint {eid} created for {name}")
    return eid


def _pg_upsert(c: Container, endpoint_id: str) -> None:
    """Insert/update row via psql to avoid taking a psycopg dep."""
    import subprocess
    name  = f"tsage-{c.name}"
    image = f"ghcr.io/kokhp/tsage-runpod-{c.name}:latest"
    sql = (
        "INSERT INTO tsage_api.runpod_endpoints "
        "(name, endpoint_id, gpu_tier, min_workers, max_workers, image_url) "
        "VALUES ($1,$2,$3,$4,$5,$6) "
        "ON CONFLICT (name) DO UPDATE SET "
        "endpoint_id = EXCLUDED.endpoint_id, "
        "gpu_tier    = EXCLUDED.gpu_tier, "
        "min_workers = EXCLUDED.min_workers, "
        "max_workers = EXCLUDED.max_workers, "
        "image_url   = EXCLUDED.image_url, "
        "updated_at  = NOW();"
    )
    subprocess.run(
        [
            "/opt/homebrew/opt/libpq/bin/psql", SUPABASE_DB_URL,
            "-v", "ON_ERROR_STOP=1",
            "-c",
            # Use psql \c-free format
            f"INSERT INTO tsage_api.runpod_endpoints "
            f"(name, endpoint_id, gpu_tier, min_workers, max_workers, image_url) "
            f"VALUES ('{name}', '{endpoint_id}', '{c.gpu_tier}', 0, 10, '{image}') "
            f"ON CONFLICT (name) DO UPDATE SET "
            f"endpoint_id=EXCLUDED.endpoint_id, "
            f"gpu_tier=EXCLUDED.gpu_tier, "
            f"min_workers=EXCLUDED.min_workers, "
            f"max_workers=EXCLUDED.max_workers, "
            f"image_url=EXCLUDED.image_url, "
            f"updated_at=NOW();"
        ],
        check=True, capture_output=True, text=True,
    )
    print(f"  row upserted for {name}")


def main() -> int:
    results: list[tuple[str, str, str | None]] = []  # (name, status, error)
    for c in CONTAINERS:
        print(f"\n=== {c.name} ({c.gpu_tier}) ===")
        try:
            tid = _create_template(c)
            eid = _create_endpoint(c, tid)
            _pg_upsert(c, eid)
            results.append((c.name, "ok", eid))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((c.name, "error", str(e)))

    print("\n\n=== SUMMARY ===")
    ok   = [r for r in results if r[1] == "ok"]
    bad  = [r for r in results if r[1] == "error"]
    print(f"created/reused: {len(ok)} / {len(CONTAINERS)}")
    for n, _, info in ok:
        print(f"  ok    tsage-{n} -> {info}")
    for n, _, info in bad:
        print(f"  FAIL  tsage-{n}: {info}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
