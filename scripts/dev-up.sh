#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$RUN_DIR/logs"
DJANGO_PID_FILE="$RUN_DIR/django.pid"
CELERY_PID_FILE="$RUN_DIR/celery.pid"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_docker_ready() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but the Docker daemon is not running." >&2
    echo "Start Docker Desktop or the Docker service, then rerun ./run.sh." >&2
    exit 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose is not available. Install the docker compose plugin and rerun ./run.sh." >&2
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-120}"
  local delay="${3:-2}"
  local i
  for ((i=1; i<=attempts; i+=1)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  echo "Timed out waiting for $url" >&2
  exit 1
}

wait_for_postgres() {
  local attempts="${1:-60}"
  local delay="${2:-2}"
  local i
  for ((i=1; i<=attempts; i+=1)); do
    if docker compose exec -T db pg_isready -U afkvision -d afkvision >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  echo "Timed out waiting for PostgreSQL" >&2
  exit 1
}

start_process() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local command="$4"

  if [[ -f "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(cat "$pid_file")"
    if kill -0 "$existing_pid" >/dev/null 2>&1; then
      echo "$name already running (pid $existing_pid)"
      return 0
    fi
    rm -f "$pid_file"
  fi

  nohup setsid bash -lc "cd '$ROOT_DIR' && exec $command" >"$log_file" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  sleep 2
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "Failed to start $name. Recent log output:" >&2
    tail -n 80 "$log_file" >&2 || true
    exit 1
  fi
  echo "Started $name (pid $pid)"
}

require_cmd docker
require_cmd curl
require_cmd uv
require_docker_ready

mkdir -p "$LOG_DIR"

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "Created .env from .env.example"
fi

cd "$ROOT_DIR"

echo "Starting Docker services..."
docker compose up -d db redis ollama inference
docker compose up -d ollama-pull || true

echo "Waiting for PostgreSQL..."
wait_for_postgres

echo "Waiting for local inference service..."
wait_for_http "http://127.0.0.1:8091/health" 180 2

echo "Installing Python dependencies..."
uv sync

echo "Running migrations..."
uv run python manage.py migrate --noinput

start_process "django" "$DJANGO_PID_FILE" "$LOG_DIR/django.log" ".venv/bin/python manage.py runserver 8001 --noreload"
start_process "celery" "$CELERY_PID_FILE" "$LOG_DIR/celery.log" ".venv/bin/celery -A config worker -l info"

echo
echo "AFK Vision is up."
echo "Web:       http://127.0.0.1:8001/"
echo "Inference: http://127.0.0.1:8091/health"
echo "Ollama:    http://127.0.0.1:11434/api/tags"
echo
echo "Logs:"
echo "  Django: $LOG_DIR/django.log"
echo "  Celery: $LOG_DIR/celery.log"
