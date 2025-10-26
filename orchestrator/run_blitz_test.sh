#!/usr/bin/env bash
#
# Quantum Temple — 12‑Hour Blitz Orchestrator
# File: orchestrator/run_blitz_test.sh
# Description:
#   Spins up the docker stack, validates health, orchestrates the 6 benchmark phases,
#   streams metrics, and writes artifacts to ./data/output_logs.
#
# Usage:
#   ./run_blitz_test.sh [-n NODES] [-d DURATION_HOURS] [--dry-run] [--no-up]
#   ENV overrides: NODES, DURATION_HOURS, COMPOSE_FILE, LOG_DIR, METRICS_PORT
#
# Dependencies:
#   - bash 4+, docker, docker compose (v2), curl, jq (optional), yq (optional)
#
set -Eeuo pipefail

####################################
# Pretty printing
####################################
NC="\033[0m"; B="\033[1m"; DIM="\033[2m"
RED="\033[31m"; GRN="\033[32m"; YEL="\033[33m"; BLU="\033[34m"; MAG="\033[35m"; CYA="\033[36m"

say()    { printf "%b[%s]%b %s\n" "$CYA" "$(date '+%F %T')" "$NC" "$*"; }
ok()     { printf "%b[OK]%b %s\n" "$GRN" "$NC" "$*"; }
warn()   { printf "%b[WARN]%b %s\n" "$YEL" "$NC" "$*"; }
err()    { printf "%b[ERR]%b %s\n" "$RED" "$NC" "$*" >&2; }

####################################
# Defaults & CLI
####################################
NODES="${NODES:-8}"
DURATION_HOURS="${DURATION_HOURS:-12}"
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
STACK_NAME="${STACK_NAME:-quantum-temple}"
LOG_DIR="${LOG_DIR:-data/output_logs}"
METRICS_PORT="${METRICS_PORT:-8080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"      # seconds to wait for healthy stack
PHASE_PAUSE="${PHASE_PAUSE:-10}"             # seconds between phases
DRY_RUN=false
NO_UP=false

usage() {
  cat <<USAGE
${B}Quantum Temple — 12‑Hour Blitz Orchestrator${NC}

${B}Usage:${NC}
  $0 [-n NODES] [-d DURATION_HOURS] [--dry-run] [--no-up]
  Environment overrides: NODES, DURATION_HOURS, COMPOSE_FILE, LOG_DIR, METRICS_PORT

${B}Examples:${NC}
  NODES=8 DURATION_HOURS=12 $0
  $0 -n 16 -d 6 --dry-run

${B}Phases:${NC}
  1) Warm-up / Gap Closure
  2) Zeta Integration
  3) Oracle Symbiosis
  4) Fractal Expansion
  5) Stress & Recovery
  6) Final Analysis

USAGE
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--nodes) NODES="${2:-}"; shift 2;;
    -d|--duration) DURATION_HOURS="${2:-}"; shift 2;;
    --dry-run) DRY_RUN=true; shift;;
    --no-up) NO_UP=true; shift;;
    -h|--help) usage; exit 0;;
    *) err "Unknown arg: $1"; usage; exit 1;;
  esac
done

####################################
# Paths & setup
####################################
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$LOG_DIR"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
ARTIFACTS="$RUN_DIR/artifacts"
mkdir -p "$ARTIFACTS"

LOG_FILE="$RUN_DIR/blitz.log"
touch "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

say "Run ID: $RUN_ID"
say "Nodes: $NODES | Duration: ${DURATION_HOURS}h | Compose: $COMPOSE_FILE"
say "Logs: $RUN_DIR"

####################################
# Preflight checks
####################################
need() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing dependency: $1"; exit 1; }
}
need docker
need bash
command -v jq >/dev/null 2>&1 || warn "jq not found; JSON parsing will be limited"
command -v yq >/dev/null 2>&1 || warn "yq not found; YAML parsing will be limited"
command -v curl >/dev/null 2>&1 || warn "curl not found; metrics pulls disabled"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  err "Compose file not found: $COMPOSE_FILE"
  exit 1
fi

####################################
# Cleanup on exit
####################################
cleanup() {
  code=$?
  if [[ $code -ne 0 ]]; then err "Aborting with status $code"; fi
  say "Collecting docker diagnostics..."
  docker ps -a > "$RUN_DIR/docker_ps.txt" || true
  docker compose -f "$COMPOSE_FILE" logs > "$RUN_DIR/docker_compose_logs.txt" || true
  ok "Artifacts saved: $RUN_DIR"
}
trap cleanup EXIT

####################################
# Stack bring-up
####################################
compose_up() {
  say "Launching docker compose stack (scale qudit-core=$NODES)"
  $DRY_RUN && { warn "DRY-RUN: skipping compose up"; return; }
  docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
  # scale qudit-core if service exists
  if docker compose -f "$COMPOSE_FILE" ps | grep -q "qudit-core"; then
    docker compose -f "$COMPOSE_FILE" up -d --scale qudit-core="$NODES"
  fi
}

wait_healthy() {
  $DRY_RUN && { warn "DRY-RUN: skipping health wait"; return; }
  say "Waiting up to ${HEALTH_TIMEOUT}s for metrics endpoint :${METRICS_PORT} ..."
  t0=$(date +%s)
  while true; do
    if curl -fsS "http://localhost:${METRICS_PORT}/health" >/dev/null 2>&1; then
      ok "Metrics endpoint is healthy"
      break
    fi
    if (( $(date +%s) - t0 > HEALTH_TIMEOUT )); then
      err "Timed out waiting for health endpoint"
      docker compose -f "$COMPOSE_FILE" ps
      exit 1
    fi
    sleep 3
  done
}

####################################
# Phase helpers
####################################
phase_banner() {
  local num="$1"; local name="$2"
  echo -e "\n${B}${BLU}=== Phase ${num}: ${name} ===${NC}\n"
}

metrics_snapshot() {
  local tag="$1"
  $DRY_RUN && return 0
  if curl -fsS "http://localhost:${METRICS_PORT}/metrics" -o "$RUN_DIR/metrics_${tag}.prom" ; then
    ok "Metrics snapshot saved: metrics_${tag}.prom"
  else
    warn "Could not fetch metrics for tag: $tag"
  fi
}

run_python_hook() {
  local hook="$1"; shift || true
  # Hook attempts: python modules or scripts (optional)
  for candidate in \
      "python3 -m src.analysis_tools.${hook}" \
      "python3 src/${hook}.py" \
      "python3 src/core/${hook}.py"; do
    if $DRY_RUN; then
      warn "DRY-RUN: would run: $candidate $*"
      return 0
    fi
    if eval "$candidate $*" >/dev/null 2>&1; then
      ok "Hook ${hook} executed"
      return 0
    fi
  done
  warn "No hook found for: ${hook} (continuing)"
  return 0
}

####################################
# Orchestration
####################################
main() {
  $NO_UP || compose_up
  $NO_UP || wait_healthy

  start_ts=$(date +%s)

  # Phase 1 — Warm-up
  phase_banner 1 "Warm-up / Gap Closure"
  run_python_hook warmup --eta 0.005
  metrics_snapshot "phase1"
  sleep "$PHASE_PAUSE"

  # Phase 2 — Zeta Integration
  phase_banner 2 "Zeta Integration"
  run_python_hook zeta --eta 0.01
  metrics_snapshot "phase2"
  sleep "$PHASE_PAUSE"

  # Phase 3 — Oracle Symbiosis
  phase_banner 3 "Oracle Symbiosis"
  run_python_hook oracle --eta 0.015
  metrics_snapshot "phase3"
  sleep "$PHASE_PAUSE"

  # Phase 4 — Fractal Expansion
  phase_banner 4 "Fractal Expansion"
  run_python_hook fractal --eta 0.02
  metrics_snapshot "phase4"
  sleep "$PHASE_PAUSE"

  # Phase 5 — Stress & Recovery
  phase_banner 5 "Stress Test / Recovery"
  run_python_hook stress --spikes 3 --partition true
  metrics_snapshot "phase5"
  sleep "$PHASE_PAUSE"

  # Phase 6 — Final Analysis
  phase_banner 6 "Final Analysis"
  run_python_hook analyze --export "$ARTIFACTS/report.json"
  metrics_snapshot "phase6"

  end_ts=$(date +%s)
  elapsed=$(( end_ts - start_ts ))
  say "Blitz complete in ${elapsed}s (~$((elapsed/3600))h)."

  # Summaries
  docker compose -f "$COMPOSE_FILE" ps > "$RUN_DIR/compose_ps.txt" || true
  docker stats --no-stream > "$RUN_DIR/docker_stats.txt" || true

  ok "All done. Artifacts: $RUN_DIR"
}

main "$@"
