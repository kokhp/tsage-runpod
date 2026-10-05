#!/usr/bin/env bash
# Deploy wrapper for dialogue-en. See ../_deploy_template.sh for the real work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../_deploy_template.sh"
tsage_deploy "dialogue-en" "NVIDIA GeForce RTX 4090" 0 3 120 600000
