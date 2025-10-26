#!/usr/bin/env bash
set -Eeuo pipefail

# CLOUD: hetzner | aws | gcp
CLOUD="${CLOUD:-hetzner}"
NODES="${NODES:-8}"
DURATION_HOURS="${DURATION_HOURS:-12}"
INSTANCE_TYPE="${INSTANCE_TYPE:-cpx41}" # hetzner default
NETWORK_NAME="${NETWORK_NAME:-quantum-mesh}"

# Load secrets (gitignored)
[[ -f ".env.secrets" ]] && source .env.secrets

mask(){ local s="${1:-}"; [[ -z "$s" ]] && echo "unset" || echo "${s:0:3}***${s:(-3)}"; }

create_hetzner(){
  REQUIRED=(HETZNER_TOKEN SSH_KEY_NAME)
  for v in "${REQUIRED[@]}"; do [[ -z "${!v:-}" ]] && { echo "Missing $v"; exit 1; }; done
  command -v hcloud >/dev/null || { echo "hcloud CLI missing"; exit 1; }

  hcloud network describe "$NETWORK_NAME" >/dev/null 2>&1 || \
    hcloud network create --name "$NETWORK_NAME" --ip-range 10.42.0.0/16

  for i in $(seq 1 "$NODES"); do
    NAME="qudit-node-$i"
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
}

create_aws(){
  REQUIRED=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION SSH_KEY_NAME)
  for v in "${REQUIRED[@]}"; do [[ -z "${!v:-}" ]] && { echo "Missing $v"; exit 1; }; done
  command -v aws >/dev/null || { echo "aws CLI missing"; exit 1; }

  AMI="${AMI:-ami-0e2c8caa4b6378d8c}" # Ubuntu 22.04 example; override as needed
  SG="${SG_ID:-sg-xxxxxxxx}"          # Pre-created security group
  SUBNET="${SUBNET_ID:-subnet-xxxxxx}"

  for i in $(seq 1 "$NODES"); do
    NAME="qudit-node-$i"
    aws ec2 run-instances \
      --image-id "$AMI" \
      --count 1 \
      --instance-type "${INSTANCE_TYPE:-c6i.2xlarge}" \
      --key-name "$SSH_KEY_NAME" \
      --security-group-ids "$SG" \
      --subnet-id "$SUBNET" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" >/dev/null &
  done
  wait
}

create_gcp(){
  REQUIRED=(GOOGLE_PROJECT GOOGLE_COMPUTE_ZONE SSH_KEY_PATH)
  for v in "${REQUIRED[@]}"; do [[ -z "${!v:-}" ]] && { echo "Missing $v"; exit 1; }; done
  command -v gcloud >/dev/null || { echo "gcloud missing"; exit 1; }

  IMAGE="${IMAGE:-ubuntu-2204-jammy-v20240702}" # example
  for i in $(seq 1 "$NODES"); do
    NAME="qudit-node-$i"
    gcloud compute instances create "$NAME" \
      --project "$GOOGLE_PROJECT" \
      --zone "$GOOGLE_COMPUTE_ZONE" \
      --machine-type "${INSTANCE_TYPE:-c2-standard-8}" \
      --image-family ubuntu-2204-lts \
      --image-project ubuntu-os-cloud \
      --metadata-from-file user-data=infrastructure/cloud-init.yaml >/dev/null &
  done
  wait
}

echo "🚀 CLOUD=$CLOUD nodes=$NODES duration=${DURATION_HOURS}h"
case "$CLOUD" in
  hetzner) create_hetzner ;;
  aws)     create_aws ;;
  gcp)     create_gcp ;;
  *) echo "Unknown CLOUD=$CLOUD"; exit 1;;
esac

echo "🔧 Configuring stack via Ansible…"
ansible-playbook -i infrastructure/inventory.yml infrastructure/deploy_stack.yml

echo "🌀 Starting Blitz…"
./orchestrator/run_blitz_test.sh &

echo "⏰ Auto-teardown in ${DURATION_HOURS}h"
sleep "$(( DURATION_HOURS * 3600 ))" || true
./orchestrator/collect_results.sh || true
./orchestrator/teardown.sh || true
echo "✅ Done."
