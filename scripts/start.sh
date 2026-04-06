#!/usr/bin/env bash
set -euo pipefail

echo "[start.sh] Starting runpod serverless worker..."

exec uv run python -m worker.main
