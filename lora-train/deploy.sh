#!/usr/bin/env bash
# Deploy wrapper for lora-train. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "lora-train" "NVIDIA L40S" 0 2 1800 600000
