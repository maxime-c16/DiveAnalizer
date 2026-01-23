#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <input_video>"
  exit 2
fi

INPUT="$1"
PORT=8765
URL="http://localhost:${PORT}/dives/review_gallery.html"

# Start diveanalyzer in background
echo "Starting diveanalyzer for ${INPUT}..."
diveanalyzer process "${INPUT}" -o ./dives --enable-server --server-port ${PORT} &
DA_PID=$!

echo "Waiting for server to become available at ${URL}..."
until curl -sSf "${URL}" >/dev/null 2>&1; do sleep 0.5; done

echo "Server up — running automation script"
python3 scripts/auto_accept.py "${URL}"

sleep 1
if ps -p ${DA_PID} >/dev/null 2>&1; then
  echo "diveanalyzer still running (killing ${DA_PID})"
  kill ${DA_PID} || true
fi

echo "Done."
