#!/usr/bin/env bash
# Deploy wrapper for agents-llm-big. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "agents-llm-big" "NVIDIA A100 80GB PCIe" 0 2 180 600000
