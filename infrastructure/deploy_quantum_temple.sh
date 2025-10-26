#!/usr/bin/env bash
set -Eeuo pipefail

# === Required secrets and tooling ===
REQUIRED=(HETZNER_TOKEN SSH_KEY_NAME)
for v in "${REQUIRED[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "❌ ERROR: Missing required env var: $v" >&2
    echo "   Provide via environment or .env.secrets (gitignored)" >&2
    exit 1
  fi
done
command -v hcloud >/dev/null || { echo "❌ hcloud CLI missing"; exit 1; }
command -v ansible-playbook >/dev/null || { echo "❌ ansible missing"; exit 1; }

# === Optional: bulk-load .env.secrets (never commit this file) ===
if [[ -f ".env.secrets" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env.secrets
  set +o allexport
fi

# Mask secrets in output
mask() { local s="${1:-}"; echo "${s:0:3}***${s:(-3)}"; }
echo "🔐 HETZNER_TOKEN: $(mask "$HETZNER_TOKEN")"
echo "🔑 SSH_KEY_NAME: $SSH_KEY_NAME"

NODES="${NODES:-8}"
INSTANCE_TYPE="${INSTANCE_TYPE:-cpx41}"
DURATION_HOURS="${DURATION_HOURS:-12}"
NETWORK_NAME="${NETWORK_NAME:-quantum-mesh}"

echo "🚀 Deploying: nodes=$NODES type=$INSTANCE_TYPE duration=${DURATION_HOURS}h"

# Create network if absent
hcloud network describe "$NETWORK_NAME" >/dev/null 2>&1 || \
  hcloud network create --name "$NETWORK_NAME" --ip-range 10.42.0.0/16

# Spin up nodes
for i in $(seq 1 "$NODES"); do
  NAME="qudit-node-$i"
  echo "🛰  $NAME"
  hcloud server create \
    --name "$NAME" \
    --type "$INSTANCE_TYPE" \
    --image ubuntu-22.04 \
    --ssh-key "$SSH_KEY_NAME" \
    --network "$NETWORK_NAME" \
    --user-data-from-file infrastructure/cloud-init.yaml \
    --no-start=false >/dev/null &
done
wait

# Configure stack
echo "🔧 Ansible provisioning…"
ansible-playbook -i infrastructure/inventory.yml infrastructure/deploy_stack.yml

# Start Blitz and background
echo "🌀 Blitz starting…"
./orchestrator/run_blitz_test.sh &

# Schedule teardown
echo "⏰ Auto-teardown in ${DURATION_HOURS}h"
sleep "$(( DURATION_HOURS * 3600 ))" || true

echo "🧹 Collect & teardown"
./orchestrator/collect_results.sh || true
./orchestrator/teardown.sh || true

echo "✅ Done."
