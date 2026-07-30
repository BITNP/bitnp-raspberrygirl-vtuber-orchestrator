#!/usr/bin/env bash
set -Eeuo pipefail

ORCHESTRATOR_PATH="${ORCHESTRATOR_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIBLING_ROOT="${SIBLING_ROOT:-$(dirname "$ORCHESTRATOR_PATH")}" 
FRONTEND_FREEZE_BASELINE="${FRONTEND_FREEZE_BASELINE:-HEAD}"

if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--sibling-root" ]]; then
    exit 2
  fi
  SIBLING_ROOT="$2"
fi

python "$ORCHESTRATOR_PATH/scripts/verify_frontend_freeze.py" --frontend-path "$SIBLING_ROOT/bitnp-raspberrygirl-vtuber-frontend" --baseline "$FRONTEND_FREEZE_BASELINE"
python "$ORCHESTRATOR_PATH/scripts/verify_protocol_schema.py"
python "$ORCHESTRATOR_PATH/scripts/verify_topology.py" --sibling-root "$SIBLING_ROOT"
python "$ORCHESTRATOR_PATH/scripts/verify_vtuber_contract.py" --frontend-path "$SIBLING_ROOT/bitnp-raspberrygirl-vtuber-frontend"
