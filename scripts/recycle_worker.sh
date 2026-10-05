#!/usr/bin/env bash
# Force a RunPod serverless endpoint to pull the latest image by cycling
# workersMax 0 -> 1. Mirrors the dance we did for the DoRA endpoint.
set -eu
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
EID="${1:?usage: recycle_worker.sh <endpoint_id>}"

api() {
  local method="$1" ; local body="$2"
  curl -fsS -X "$method" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    "https://rest.runpod.io/v1/endpoints/$EID" \
    -d "$body"
}

echo "scaling $EID -> workersMax=0"
api PATCH '{"workersMax":0}'
echo
sleep 30
echo "scaling $EID -> workersMax=1"
api PATCH '{"workersMax":1}'
echo
echo "done"
