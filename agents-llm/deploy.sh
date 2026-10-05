#!/usr/bin/env bash
# Deploy wrapper for agents-llm. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "agents-llm" "NVIDIA RTX A5000" 1 10 30 600000
