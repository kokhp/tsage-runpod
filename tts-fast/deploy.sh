#!/usr/bin/env bash
# Deploy wrapper for tts-fast. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "tts-fast" "NVIDIA RTX A4000" 0 10 60 600000
