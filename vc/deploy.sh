#!/usr/bin/env bash
# Deploy wrapper for vc. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "vc" "NVIDIA RTX A4000" 0 6 60 600000
