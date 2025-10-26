#!/usr/bin/env bash
set -Eeuo pipefail
#
# Quantum Temple — 12-Hour Blitz Deployment
# provisions 8-node test cluster on Hetzner and launches the orchestrator stack
# safely handles secrets and auto-teardown after 12 hours
#

# ────────────────────────────────────────────────
# Env & sanity checks
# ────────────────────────────────────────────────
REQUIRED=(HETZNER_TOKEN SSH_KEY_NAME)
for v in "${REQUIRED[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "❌ ERROR: Missing required env var: $v" >&2
    exit 1
  fi
done

if [[ -f ".env.secrets" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env.secrets | xargs -0 || true)
fi

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
NODES=${NODES:-8}
INSTANCE_TYPE=${INSTANCE_TYPE:-cpx41}
NETWORK_NAME=${NETWORK_NAME:-quantum-mesh}
DURATION=${DURATION:-12h}
CONFIG_FILE=${CONFIG_FILE:-infrastructure/cloud-init.yaml}

echo "🚀 Deploying Quantum Temple Blitz: ${NODES}×${INSTANCE_TYPE}"

# ────────────────────────────────────────────────
# Create network if needed
# ────────────────────────────────────────────────
if ! hcloud network describe "$NETWORK_NAME" >/dev/null 2>&1; then
  echo "🧩 Creating network ${NETWORK_NAME}"
  hcloud network create --name "$NETWORK_NAME" --ip-range 10.42.0.0/16
fi

# ────────────────────────────────────────────────
# Launch servers
# ────────────────────────────────────────────────
for i in $(seq 1 "$NODES"); do
  echo "🌐 Spawning node $i"
  hcloud server create \
    --name "qudit-node-$i" \
    --type "$INSTANCE_TYPE" \
    --image ubuntu-22.04 \
    --ssh-key "$SSH_KEY_NAME" \
    --network "$NETWORK_NAME" \
    --user-data-from-file "$CONFIG_FILE" &
done
wait
echo "✅ All nodes created."

# ────────────────────────────────────────────────
# Configure via Ansible
# ────────────────────────────────────────────────
echo "🧠 Configuring stack via Ansible..."
ansible-playbook -i infrastructure/inventory.yml infrastructure/deploy_stack.yml

# ────────────────────────────────────────────────
# Run orchestrator
# ────────────────────────────────────────────────
echo "▶ Starting 12-Hour Blitz benchmark..."
./orchestrator/run_blitz_test.sh

# ────────────────────────────────────────────────
# Auto-destruct timer
# ────────────────────────────────────────────────
echo "⏰ Auto-termination scheduled after ${DURATION}"
sleep 43200  # 12h in seconds

echo "💀 Tearing down..."
./orchestrator/collect_results.sh || true
./orchestrator/teardown.sh || true
echo "🏁 Blitz completed and cleaned up."
