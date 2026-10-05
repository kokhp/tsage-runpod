#!/usr/bin/env bash
# Deploy wrapper for asr-omnilingual. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "asr-omnilingual" "NVIDIA RTX A5000" 0 6 90 600000
