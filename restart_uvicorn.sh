#!/usr/bin/env bash
# ------------------- Sovereign Secrets -------------------
export GEMINI_API_KEY="AIzaSyDY4urZvNlqSWaKasnQGWgSgk3P2Rfcgag"
export MODEL_ID="microsoft/DialoGPT-small"
export TEMPERATURE="0.1"

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/venv/bin/python"
if [ ! -x "$PY" ]; then PY=python; fi
PID_FILE="$DIR/uvicorn.pid"
LOG="$DIR/uvicorn.log"
PORT=8000

stop() {
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "Killing $pid"
      kill -9 "$pid" || true
    fi
    rm -f "$PID_FILE"
  fi
  # also kill any process listening on PORT
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t || true)
    if [ -n "$pids" ]; then
      echo "Killing processes on port $PORT: $pids"
      for x in $pids; do kill -9 $x || true; done
    fi
  fi
}

start() {
  nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --log-level info > "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Started PID: $(cat "$PID_FILE")"
}

case "${1:-}" in
  start) stop; start ;;
  stop) stop ;;
  restart) stop; start ;;
  *) echo "Usage: $0 {start|stop|restart}"; exit 2 ;;
esac
