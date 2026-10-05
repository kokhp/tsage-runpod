#!/usr/bin/env bash
# Deploy wrapper for asr-align. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "asr-align" "NVIDIA RTX A4000" 0 8 45 600000
