#!/usr/bin/env bash
set -euo pipefail

echo "[pipeline/run.sh] starting pipeline" >&2
echo "[pipeline/run.sh] data_root=/app/data" >&2
echo "[pipeline/run.sh] ingest_url=${API_INGEST_URL:-http://localhost:8000/events/ingest}" >&2

python3 pipeline/detect.py | python3 pipeline/emit.py
