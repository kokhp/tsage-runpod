#!/usr/bin/env bash
# Deploy wrapper for asr-realtime. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "asr-realtime" "NVIDIA RTX A4000" 1 20 30 600000
