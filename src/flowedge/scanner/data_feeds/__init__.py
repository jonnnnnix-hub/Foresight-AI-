"""Multi-source data feed architecture for FlowEdge.

Ingests data from:
- Polygon.io (paid): 1m/5m/15m bars, options chains, real-time quotes
- Alpaca Markets: real-time quotes, paper trading execution
- VIX/VVIX: fear gauge for regime detection
- Market breadth: TICK, ADD, VOLD for capitulation detection
- Sentiment: Benzinga news, StockTwits social
"""
