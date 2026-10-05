#!/usr/bin/env bash
# Build + push tts-clone image and create/update its RunPod Serverless endpoint.
#
# Required env:
#   RUNPOD_API_KEY   — from ~/.claude/env/runpod.env
#   HF_TOKEN         — from ~/.claude/env/huggingface.env
#   REGISTRY         — docker image registry prefix (e.g. ghcr.io/kokhp)
#
# Optional env:
#   IMAGE_TAG        — defaults to "latest"
#   GPU_IDS          — comma list of GPU type IDs (defaults to A5000: NVIDIA RTX A5000)
#   MIN_WORKERS      — defaults to 0 (scale-to-zero)
#   MAX_WORKERS      — defaults to 10
#   IDLE_TIMEOUT     — defaults to 60 (seconds)
#   EXECUTION_TIMEOUT_MS — defaults to 600000 (10 min)
#   VOLUME_ID        — pre-existing RunPod network volume ID
set -euo pipefail

NAME="tts-clone"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- sanity --------------------------------------------------------------------
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (source ~/.claude/env/runpod.env)}"
: "${HF_TOKEN:?HF_TOKEN required (source ~/.claude/env/huggingface.env)}"
: "${REGISTRY:?REGISTRY required (e.g. ghcr.io/kokhp)}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${REGISTRY}/tsage-runpod-${NAME}:${IMAGE_TAG}"

GPU_IDS="${GPU_IDS:-NVIDIA RTX A5000}"
MIN_WORKERS="${MIN_WORKERS:-0}"
MAX_WORKERS="${MAX_WORKERS:-10}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-600000}"

echo "=== $NAME → $IMAGE ==="

# -- build + push --------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  docker buildx build --platform linux/amd64 -t "$IMAGE" --push "$HERE"
elif command -v nerdctl >/dev/null 2>&1; then
  nerdctl build --platform linux/amd64 -t "$IMAGE" "$HERE"
  nerdctl push "$IMAGE"
else
  echo "ERROR: no docker/nerdctl in PATH. Install one, then re-run."
  exit 1
fi

# -- create or update endpoint via RunPod GraphQL ------------------------------
# RunPod exposes a REST+GraphQL mix. Serverless endpoints use the GraphQL schema.
# We first look up an existing endpoint with the same name and, if present,
# update its image + scaling config; otherwise we create a new one.

GQL='https://api.runpod.io/graphql'

existing_id=$(curl -sS -X POST "$GQL" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"query{myself{serverlessEndpoints{id name}}}\"}" |
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for e in data.get('data', {}).get('myself', {}).get('serverlessEndpoints') or []:
    if e['name'] == 'tsage-$NAME':
        print(e['id'])
        break
")

ENV_JSON=$(python3 -c "
import json
print(json.dumps([
  {'key':'HF_TOKEN','value':'$HF_TOKEN'},
  {'key':'RUNPOD_VOLUME','value':'/runpod-volume'},
]))
")

if [ -n "$existing_id" ]; then
  echo "Updating existing endpoint $existing_id"
  curl -sS -X POST "$GQL" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json, os
mutation = '''
mutation UpdateEndpoint(\$input: UpdateEndpointInput!) {
  updateEndpoint(input: \$input) { id name }
}
'''
print(json.dumps({
  'query': mutation,
  'variables': {'input': {
    'id': '$existing_id',
    'imageName': '$IMAGE',
    'minWorkers': int('$MIN_WORKERS'),
    'maxWorkers': int('$MAX_WORKERS'),
    'idleTimeout': int('$IDLE_TIMEOUT'),
    'executionTimeoutMs': int('$EXECUTION_TIMEOUT_MS'),
  }}
}))
")" | python3 -m json.tool
else
  echo "Creating new endpoint tsage-$NAME"
  curl -sS -X POST "$GQL" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json, os
mutation = '''
mutation CreateEndpoint(\$input: CreateEndpointInput!) {
  createEndpoint(input: \$input) { id name }
}
'''
input_obj = {
  'name': 'tsage-$NAME',
  'imageName': '$IMAGE',
  'gpuIds': '$GPU_IDS',
  'minWorkers': int('$MIN_WORKERS'),
  'maxWorkers': int('$MAX_WORKERS'),
  'idleTimeout': int('$IDLE_TIMEOUT'),
  'executionTimeoutMs': int('$EXECUTION_TIMEOUT_MS'),
  'env': $ENV_JSON,
}
vol = os.environ.get('VOLUME_ID')
if vol:
    input_obj['networkVolumeId'] = vol
print(json.dumps({'query': mutation, 'variables': {'input': input_obj}}))
")" | python3 -m json.tool
fi

echo "=== $NAME done ==="
