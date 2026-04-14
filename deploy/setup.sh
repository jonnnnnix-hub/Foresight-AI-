#!/bin/bash
# FlowEdge Bots — First-time Fly.io deployment setup
#
# Prerequisites:
#   1. Install flyctl: curl -L https://fly.io/install.sh | sh
#   2. Login: fly auth login
#   3. Set your secrets (see below)
#
# Usage:
#   chmod +x deploy/setup.sh
#   ./deploy/setup.sh

set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "  FlowEdge Paper Trading Bots — Fly.io Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "ERROR: flyctl not found. Install it first:"
    echo "  curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "ERROR: Not logged in to Fly.io. Run: fly auth login"
    exit 1
fi

echo "Step 1: Creating Fly.io app..."
flyctl apps create flowedge-bots --org personal 2>/dev/null || echo "App already exists"

echo ""
echo "Step 2: Creating persistent volume for logs + data..."
flyctl volumes create flowedge_data --region iad --size 1 --yes 2>/dev/null || echo "Volume already exists"

echo ""
echo "Step 3: Setting secrets..."
echo ""
echo "You need to set these secrets. Run each command below"
echo "with your actual values:"
echo ""
echo "  # Market Data"
echo "  fly secrets set POLYGON_API_KEY=your_polygon_key"
echo ""
echo "  # Account 1: Scalp v2 + Volume Scalper v1"
echo "  fly secrets set ALPACA_API_KEY_ID=your_key ALPACA_API_SECRET_KEY=your_secret"
echo ""
echo "  # Account 2: Production Scanner"
echo "  fly secrets set ALPACA_PROD_KEY_ID=your_key ALPACA_PROD_SECRET_KEY=your_secret"
echo ""
echo "  # Account 3: Trident"
echo "  fly secrets set TRIDENT_ALPACA_KEY_ID=your_key TRIDENT_ALPACA_SECRET_KEY=your_secret"
echo ""
echo "  # Email Alerts (optional)"
echo "  fly secrets set ALERT_EMAIL_TO=you@gmail.com ALERT_EMAIL_FROM=you@gmail.com \\"
echo "    ALERT_SMTP_HOST=smtp.gmail.com ALERT_SMTP_PORT=587 \\"
echo "    ALERT_SMTP_USER=you@gmail.com ALERT_SMTP_PASS=your_app_password"
echo ""

read -p "Have you set all secrets? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Set your secrets first, then run: fly deploy"
    exit 0
fi

echo ""
echo "Step 4: Deploying..."
flyctl deploy --remote-only

echo ""
echo "Step 5: Verifying..."
sleep 15
flyctl ssh console -C "supervisorctl -c /etc/supervisor/conf.d/flowedge.conf status"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  Monitor logs:    fly logs"
echo "  Bot status:      fly ssh console -C 'supervisorctl status'"
echo "  Restart a bot:   fly ssh console -C 'supervisorctl restart bots:scalp_v2'"
echo "  Restart all:     fly ssh console -C 'supervisorctl restart bots:*'"
echo "  Scale memory:    fly scale memory 1024"
echo "═══════════════════════════════════════════════════════════"
