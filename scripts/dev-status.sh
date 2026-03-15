#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

show_process() {
  local name="$1"
  local pid_file="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name: running (pid $pid)"
      return
    fi
    echo "$name: stale pid file ($pid)"
    return
  fi
  echo "$name: not running"
}

cd "$ROOT_DIR"

echo "Process status"
show_process "django" "$RUN_DIR/django.pid"
show_process "celery" "$RUN_DIR/celery.pid"

echo
echo "Docker status"
docker compose ps

echo
echo "Endpoints"
curl -fsS http://127.0.0.1:8001/ >/dev/null 2>&1 && echo "web: ok" || echo "web: down"
curl -fsS http://127.0.0.1:8091/health >/dev/null 2>&1 && echo "inference: ok" || echo "inference: down"
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && echo "ollama: ok" || echo "ollama: down"
