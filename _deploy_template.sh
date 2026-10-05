#!/usr/bin/env bash
# Shared deploy template — each container's deploy.sh is a thin wrapper around this.
# Sourced via: source "$HERE/../_deploy_template.sh" and then calling tsage_deploy.
set -euo pipefail

tsage_deploy() {
  local name="$1"
  local gpu_ids="$2"
  local min_workers="${3:-0}"
  local max_workers="${4:-10}"
  local idle_timeout="${5:-60}"
  local execution_timeout_ms="${6:-600000}"
  local cpu_flavor="${7:-}"            # pass "CPU3-{N}" or leave blank for GPU
  local extra_env_json="${8:-[]}"

  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (source ~/.claude/env/runpod.env)}"
  : "${HF_TOKEN:?HF_TOKEN required (source ~/.claude/env/huggingface.env)}"
  : "${REGISTRY:?REGISTRY required (e.g. ghcr.io/kokhp)}"
  local image_tag="${IMAGE_TAG:-latest}"
  local image="${REGISTRY}/tsage-runpod-${name}:${image_tag}"
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"

  echo "=== $name → $image ==="

  if command -v docker >/dev/null 2>&1; then
    docker buildx build --platform linux/amd64 -t "$image" --push "$here"
  elif command -v nerdctl >/dev/null 2>&1; then
    nerdctl build --platform linux/amd64 -t "$image" "$here"
    nerdctl push "$image"
  else
    echo "ERROR: install docker or nerdctl first"; return 1
  fi

  local gql='https://api.runpod.io/graphql'
  local existing_id
  existing_id=$(curl -sS -X POST "$gql" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query{myself{serverlessEndpoints{id name}}}\"}" |
    python3 -c "
import json, sys, os
name = 'tsage-' + os.environ['TSAGE_NAME']
data = json.load(sys.stdin)
for e in data.get('data', {}).get('myself', {}).get('serverlessEndpoints') or []:
    if e['name'] == name:
        print(e['id']); break
" TSAGE_NAME="$name" 2>/dev/null || true)

  local env_json
  env_json=$(TSAGE_EXTRA="$extra_env_json" python3 -c "
import json, os
base = [
  {'key':'HF_TOKEN','value':os.environ['HF_TOKEN']},
  {'key':'RUNPOD_VOLUME','value':'/runpod-volume'},
]
extra = json.loads(os.environ.get('TSAGE_EXTRA','[]') or '[]')
print(json.dumps(base + extra))
")

  if [ -n "${existing_id:-}" ]; then
    echo "Updating existing endpoint $existing_id"
    TSAGE_ID="$existing_id" TSAGE_IMG="$image" TSAGE_MIN="$min_workers" \
    TSAGE_MAX="$max_workers" TSAGE_IDLE="$idle_timeout" \
    TSAGE_TIMEOUT="$execution_timeout_ms" \
    python3 -c "
import os, json, urllib.request
mutation = '''
mutation UpdateEndpoint(\$input: UpdateEndpointInput!) {
  updateEndpoint(input: \$input) { id name }
}
'''
body = json.dumps({'query': mutation, 'variables': {'input': {
  'id': os.environ['TSAGE_ID'], 'imageName': os.environ['TSAGE_IMG'],
  'minWorkers': int(os.environ['TSAGE_MIN']),
  'maxWorkers': int(os.environ['TSAGE_MAX']),
  'idleTimeout': int(os.environ['TSAGE_IDLE']),
  'executionTimeoutMs': int(os.environ['TSAGE_TIMEOUT']),
}}}).encode()
req = urllib.request.Request('https://api.runpod.io/graphql', data=body,
  headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],
           'Content-Type':'application/json'})
print(urllib.request.urlopen(req, timeout=30).read().decode())
"
  else
    echo "Creating new endpoint tsage-$name"
    TSAGE_NAME="tsage-$name" TSAGE_IMG="$image" TSAGE_GPUS="$gpu_ids" \
    TSAGE_MIN="$min_workers" TSAGE_MAX="$max_workers" \
    TSAGE_IDLE="$idle_timeout" TSAGE_TIMEOUT="$execution_timeout_ms" \
    TSAGE_ENV="$env_json" TSAGE_VOLUME="${VOLUME_ID:-}" \
    python3 -c "
import os, json, urllib.request
mutation = '''
mutation CreateEndpoint(\$input: CreateEndpointInput!) {
  createEndpoint(input: \$input) { id name }
}
'''
inp = {
  'name': os.environ['TSAGE_NAME'],
  'imageName': os.environ['TSAGE_IMG'],
  'gpuIds': os.environ['TSAGE_GPUS'],
  'minWorkers': int(os.environ['TSAGE_MIN']),
  'maxWorkers': int(os.environ['TSAGE_MAX']),
  'idleTimeout': int(os.environ['TSAGE_IDLE']),
  'executionTimeoutMs': int(os.environ['TSAGE_TIMEOUT']),
  'env': json.loads(os.environ['TSAGE_ENV']),
}
if os.environ.get('TSAGE_VOLUME'):
    inp['networkVolumeId'] = os.environ['TSAGE_VOLUME']
body = json.dumps({'query': mutation, 'variables': {'input': inp}}).encode()
req = urllib.request.Request('https://api.runpod.io/graphql', data=body,
  headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],
           'Content-Type':'application/json'})
print(urllib.request.urlopen(req, timeout=30).read().decode())
"
  fi

  echo "=== $name done ==="
}
