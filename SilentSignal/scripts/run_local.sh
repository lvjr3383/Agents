#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. uvicorn services.api.app.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
python3 -m http.server 5173 -d apps/dashboard &
DASH_PID=$!

cleanup() {
  kill "$API_PID" "$DASH_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

wait -n "$API_PID" "$DASH_PID"
