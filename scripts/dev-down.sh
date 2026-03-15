#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
DJANGO_PID_FILE="$RUN_DIR/django.pid"
CELERY_PID_FILE="$RUN_DIR/celery.pid"

stop_process() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name not running"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "Stopped $name"
  else
    echo "$name pid file was stale"
  fi
  rm -f "$pid_file"
}

cd "$ROOT_DIR"

stop_process "django" "$DJANGO_PID_FILE"
stop_process "celery" "$CELERY_PID_FILE"

echo "Stopping Docker services..."
docker compose down

echo "AFK Vision is down."
