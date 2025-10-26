#!/usr/bin/env bash
#
# Quantum Temple — 12-Hour Blitz Orchestrator
# Spins up the docker stack, validates health, orchestrates the 6 phases,
# snapshots metrics, and stores artifacts in ./data/output_logs/<RUN_ID>.
#
set -Eeuo pipefail

NC="\033[0m"; B="\033[1m"; DIM="\033[2m"
RED="\033[31m"; GRN="\033[32m"; YEL="\033[33m"; BLU="\033[34m"; CYA="\033[36m"
say(){ printf "%b[%s]%b %s\n" "$CYA" "$(date '+%F %T')" "$NC" "$*"; }
ok(){ printf "%b[OK]%b %s\n" "$GRN" "$NC" "$*"; }
warn(){ printf "%b[WARN]%b %s\n" "$YEL" "$NC" "$*"; }
err(){ printf "%b[ERR]%b %s\n" "$RED" "$NC" "$*" >&2; }

NODES="${NODES:-8}"
DURATION_HOURS="${DURATION_HOURS:-12}"
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
STACK_NAME="${STACK_NAME:-quantum-temple}"
LOG_DIR="${LOG_DIR:-data/output_logs}"
METRICS_PORT="${METRICS_PORT:-8080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
PHASE_PAUSE="${PHASE_PAUSE:-10}"
DRY_RUN=false; NO_UP=false

usage(){ cat <<USAGE
$B Quantum Temple — 12-Hour Blitz Orchestrator $NC

Usage:
  $0 [-n NODES] [-d DURATION_HOURS] [--dry-run] [--no-up]
Env:
  NODES, DURATION_HOURS, COMPOSE_FILE, LOG_DIR, METRICS_PORT, HEALTH_TIMEOUT, PHASE_PAUSE
USAGE
}

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

need(){ command -v "$1" >/dev/null 2>&1 || { err "Missing dependency: $1"; exit 1; }; }
need docker; need bash
command -v curl >/dev/null 2>&1 || warn "curl not found; metrics pulls disabled"
command -v jq >/dev/null 2>&1 || warn "jq not found; JSON parsing limited"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$LOG_DIR"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$LOG_DIR/$RUN_ID"; mkdir -p "$RUN_DIR" "$RUN_DIR/artifacts"
LOG_FILE="$RUN_DIR/blitz.log"; touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

say "Run ID: $RUN_ID"
say "Nodes: $NODES | Duration: ${DURATION_HOURS}h | Compose: $COMPOSE_FILE"
say "Logs: $RUN_DIR"

cleanup(){
  code=$?
  if [[ $code -ne 0 ]]; then err "Aborting with status $code"; fi
  say "Collecting docker diagnostics..."
  docker ps -a > "$RUN_DIR/docker_ps.txt" || true
  docker compose -f "$COMPOSE_FILE" logs > "$RUN_DIR/docker_compose_logs.txt" || true
  ok "Artifacts saved: $RUN_DIR"
}
trap cleanup EXIT

compose_up(){
  say "Launching docker compose (scale qudit-core=$NODES)"
  $DRY_RUN && { warn "DRY-RUN: skipping compose up"; return; }
  docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
  if docker compose -f "$COMPOSE_FILE" ps | grep -q "qudit-core"; then
    docker compose -f "$COMPOSE_FILE" up -d --scale qudit-core="$NODES"
  fi
}

wait_healthy(){
  $DRY_RUN && { warn "DRY-RUN: skipping health wait"; return; }
  say "Waiting up to ${HEALTH_TIMEOUT}s for http://localhost:${METRICS_PORT}/health"
  t0=$(date +%s)
  while true; do
    if curl -fsS "http://localhost:${METRICS_PORT}/health" >/dev/null 2>&1; then
      ok "Metrics endpoint healthy"
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

phase_banner(){ echo -e "\n${B}${BLU}=== Phase $1: $2 ===${NC}\n"; }
metrics_snapshot(){
  local tag="$1"
  $DRY_RUN && return 0
  curl -fsS "http://localhost:${METRICS_PORT}/metrics" -o "$RUN_DIR/metrics_${tag}.prom" \
    && ok "Metrics snapshot: metrics_${tag}.prom" || warn "Failed metrics snapshot: ${tag}"
}
run_hook(){
  local hook="$1"; shift || true
  for cmd in \
    "python3 -m src.phase_hooks.${hook}" \
    "python3 src/phase_hooks/${hook}.py" \
    "python3 src/${hook}.py" \
    "python3 src/core/${hook}.py" ; do
      if $DRY_RUN; then warn "DRY-RUN: would run: $cmd $*"; return 0; fi
      if eval "$cmd $*" >/dev/null 2>&1; then ok "Hook ${hook} executed"; return 0; fi
  done
  warn "No hook found: ${hook} (continuing)"
}

main(){
  $NO_UP || compose_up
  $NO_UP || wait_healthy
  start_ts=$(date +%s)

  phase_banner 1 "Warm-up / Gap Closure";    run_hook warmup --eta 0.005;  metrics_snapshot "phase1"; sleep "$PHASE_PAUSE"
  phase_banner 2 "Zeta Integration";         run_hook zeta --eta 0.01;     metrics_snapshot "phase2"; sleep "$PHASE_PAUSE"
  phase_banner 3 "Oracle Symbiosis";         run_hook oracle --eta 0.015;  metrics_snapshot "phase3"; sleep "$PHASE_PAUSE"
  phase_banner 4 "Fractal Expansion";        run_hook fractal --eta 0.02;  metrics_snapshot "phase4"; sleep "$PHASE_PAUSE"
  phase_banner 5 "Stress & Recovery";        run_hook stress --spikes 3;   metrics_snapshot "phase5"; sleep "$PHASE_PAUSE"
  phase_banner 6 "Final Analysis";           run_hook analyze --export "$RUN_DIR/artifacts/report.json"; metrics_snapshot "phase6"

  end_ts=$(date +%s); elapsed=$(( end_ts - start_ts ))
  say "Blitz complete in ${elapsed}s (~$((elapsed/3600))h)."
  docker compose -f "$COMPOSE_FILE" ps > "$RUN_DIR/compose_ps.txt" || true
  docker stats --no-stream > "$RUN_DIR/docker_stats.txt" || true
  ok "All done. Artifacts: $RUN_DIR"
}
main "$@"
