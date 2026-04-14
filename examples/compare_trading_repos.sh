#!/usr/bin/env bash
# FlowEdge MVP Comparison — vectorbt vs NautilusTrader vs Freqtrade
#
# Prerequisites:
#   1. Python 3.12+ with venv activated
#   2. ANTHROPIC_API_KEY set in .env or environment
#   3. pip install -e ".[dev]"
#
# Usage:
#   ./examples/compare_trading_repos.sh

set -euo pipefail

echo "=== FlowEdge Repo Comparison ==="
echo "Analyzing: vectorbt, NautilusTrader, Freqtrade"
echo ""

flowedge analyze \
    https://github.com/polakowo/vectorbt \
    https://github.com/nautechsystems/nautilus_trader \
    https://github.com/freqtrade/freqtrade \
    --output-dir ./output \
    --log-level INFO

echo ""
echo "=== Reports saved to ./output/ ==="
echo "JSON and Markdown reports available."
