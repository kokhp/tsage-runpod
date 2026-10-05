#!/usr/bin/env bash
# Deploy wrapper for mt. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "mt" "NVIDIA RTX A5000" 0 6 60 600000
