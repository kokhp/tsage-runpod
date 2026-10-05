#!/usr/bin/env bash
# Deploy wrapper for separator-cpu. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "separator-cpu" "" 1 1 5 600000
