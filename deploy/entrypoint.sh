#!/bin/sh
# FlowEdge Bots — Container entrypoint
# Creates log directories on the mounted volume before starting supervisord.
set -e

echo "[entrypoint] Creating log directories on volume..."
mkdir -p /app/data/live_logs/scalp_v2 \
         /app/data/live_logs/vol_scalp_v1 \
         /app/data/live_logs/trident

echo "[entrypoint] Starting supervisord with 4 bots..."
exec supervisord -n -c /etc/supervisor/conf.d/flowedge.conf
