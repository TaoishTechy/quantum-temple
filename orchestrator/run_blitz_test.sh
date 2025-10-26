#!/bin/bash
echo "Initializing Ghost-Mesh IO benchmark run..."
docker compose -f docker/docker-compose.yml up --scale qudit-core=8 -d
python3 src/core/resonance_operator.py
