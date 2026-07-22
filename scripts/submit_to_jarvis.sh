#!/usr/bin/env bash
# Template script: push Docker image and submit a job to Jarvis Lab.
# Customize for your Jarvis/registry CLI.

set -euo pipefail

REGISTRY=${REGISTRY:-"<your-registry>"}
IMAGE=${IMAGE:-"core-rl-agent:latest"}
PROJECT=${JARVIS_PROJECT:-"<project-id>"}

# Build image (assumes tar unpacked or Dockerfile present)
docker build -t ${REGISTRY}/${IMAGE} .

# Push image
docker push ${REGISTRY}/${IMAGE}

# Submit job (placeholder - replace with Jarvis CLI/API call)
# jarvis job submit --project ${PROJECT} --image ${REGISTRY}/${IMAGE} --gpus 1 --cpus 8 --memory 64GB --command "python trainers.py --data-dir /workspace/data/precomputed --device cuda"

echo "Image pushed to ${REGISTRY}/${IMAGE}. Use Jarvis CLI to submit a job using this image."
