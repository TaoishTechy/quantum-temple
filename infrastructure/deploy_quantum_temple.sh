#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Quantum Temple – Automated Deployment Script
# Deploys multi-node cluster for 12-Hour Blitz or research runs.
# Includes security checks, secret isolation, and reproducibility.
# ──────────────────────────────────────────────────────────────
set -Eeuo pipefail

# Required environment variables
REQUIRED=(HETZNER_TOKEN SSH_KEY_NAME)
for v in "${REQUIRED[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "❌ ERROR: Missing required environment variable: $v" >&2
    exit 1
  fi
done

# Optional .env.secrets file (gitignored)
if [[ -f ".env.secrets" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env.secrets | xargs -0 || true)
fi

NODES="${NODES:-8}"
INSTANCE_TYPE="${INSTANCE_TYPE:-cpx41}"
DURATION_HOURS="${DURATION_HOURS:-12}"
NETWORK_NAME="quantum-mesh"
SSH_KEY="${SSH_KEY_NAME}"

echo "🚀 Deploying Quantum Temple Cluster"
echo "• Nodes: $NODES"
echo "• Type: $INSTANCE_TYPE"
echo "• Duration: $DURATION_HOURS hours"
echo "─────────────────────────────────────"

# Create private network (if missing)
if ! hcloud network describe "$NETWORK_NAME" >/dev/null 2>&1; then
  echo "🧩 Creating network $NETWORK_NAME..."
  hcloud network create --name "$NETWORK_NAME" --ip-range 10.42.0.0/16
fi

# Spin up nodes
for i in $(seq 1 "$NODES"); do
  NAME="qudit-node-$i"
  echo "🛰  Launching $NAME..."
  hcloud server create \
    --name "$NAME" \
    --type "$INSTANCE_TYPE" \
    --image ubuntu-22.04 \
    --ssh-key "$SSH_KEY" \
    --network "$NETWORK_NAME" \
    --user-data-from-file infrastructure/cloud-init.yaml \
    --no-start=false >/dev/null &
done
wait

# Configure cluster using Ansible
echo "🔧 Configuring stack..."
ansible-playbook -i infrastructure/inventory.yml infrastructure/deploy_stack.yml

# Kick off Blitz run
echo "🌀 Starting Blitz test..."
./orchestrator/run_blitz_test.sh &

# Monitor run duration and schedule cleanup
echo "⏰ Run will terminate in ${DURATION_HOURS}h"
sleep "$(( DURATION_HOURS * 3600 ))"

echo "🧹 Collecting results and tearing down..."
./orchestrator/collect_results.sh
./orchestrator/teardown.sh

echo "✅ Deployment complete."
