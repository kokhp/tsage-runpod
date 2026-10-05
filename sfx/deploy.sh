#!/usr/bin/env bash
# Deploy wrapper for sfx. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "sfx" "NVIDIA RTX A4000" 0 4 60 600000
