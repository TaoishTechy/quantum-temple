#!/bin/bash
set -e
echo "Deploying Quantum Temple 12-Hour Blitz..."
NODES=8
INSTANCE_TYPE=cpx41
DURATION=12h
hcloud network create --name quantum-mesh --ip-range 10.42.0.0/16
for i in $(seq 1 $NODES); do
  hcloud server create --name qudit-node-$i \
    --type $INSTANCE_TYPE \
    --image ubuntu-22.04 \
    --ssh-key ~/.ssh/quantum_temple.pub \
    --network quantum-mesh \
    --user-data-from-file cloud-init.yaml &
done
wait
ansible-playbook -i inventory.yml deploy_stack.yml
./orchestrator/run_blitz_test.sh
sleep 43200
./orchestrator/collect_results.sh && ./orchestrator/teardown.sh
