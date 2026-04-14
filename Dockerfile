# FlowEdge Paper Trading Bots — Production Docker Image
# Runs 4 concurrent scanners via supervisord:
#   1. Scalp v2 (Account 1)
#   2. Volume Scalper v1 (Account 1)
#   3. Production Scanner — precision/hybrid/rapid (Account 2)
#   4. Trident ETF 0DTE scalper (Account 3)

FROM python:3.12-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps — install from pyproject.toml + scanner extras
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e .
# Additional runtime deps not in pyproject.toml
RUN pip install --no-cache-dir python-dotenv boto3

# Copy scripts, configs, deploy
COPY scripts/ scripts/
COPY configs/ configs/
COPY deploy/ deploy/

# Supervisord config + entrypoint
COPY deploy/supervisord.conf /etc/supervisor/conf.d/flowedge.conf
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Health check — verify at least one bot process is running
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD supervisorctl -c /etc/supervisor/conf.d/flowedge.conf status | grep -q RUNNING || exit 1

# Persistent volume for logs and cached data
VOLUME ["/app/data"]

# Entrypoint creates log dirs on the mounted volume, then starts supervisord
CMD ["/app/entrypoint.sh"]
