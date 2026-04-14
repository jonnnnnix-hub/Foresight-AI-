"""Scanner engine codenames — because every edge deserves a name."""

# Engine codenames
SPECTER = "SPECTER"      # UOA Scanner — sees invisible institutional flow
ORACLE = "ORACLE"        # IV Rank — predicts mispriced premiums
SENTINEL = "SENTINEL"    # Catalyst Scanner — watches for binary events
VORTEX = "VORTEX"        # GEX — maps gamma forces that move price
PULSE = "PULSE"          # Momentum — reads the heartbeat of price action
ARCHITECT = "ARCHITECT"  # Contract Selector — designs optimal structures
NEXUS = "NEXUS"          # Composite Scorer — fuses all signals into one
CIPHER = "CIPHER"        # AI Interpreter — decodes what the data means
PHANTOM = "PHANTOM"      # Backtester — ghost trades through history

ENGINE_CODENAMES: dict[str, str] = {
    "uoa": SPECTER,
    "iv_rank": ORACLE,
    "catalyst": SENTINEL,
    "gex": VORTEX,
    "momentum": PULSE,
    "selector": ARCHITECT,
    "scorer": NEXUS,
    "interpreter": CIPHER,
    "backtest": PHANTOM,
}

ENGINE_DESCRIPTIONS: dict[str, str] = {
    SPECTER: "Detects invisible institutional flow — sweeps, blocks, dark pool prints",
    ORACLE: "Predicts when option premiums are historically mispriced",
    SENTINEL: "Watches the horizon for earnings, insider moves, and binary catalysts",
    VORTEX: "Maps dealer gamma forces that pin, repel, or accelerate price",
    PULSE: "Reads multi-timeframe momentum — RSI, MACD, trend alignment",
    ARCHITECT: "Designs optimal strike, expiry, and structure for each play",
    NEXUS: "Fuses all signal dimensions into a single conviction score",
    CIPHER: "AI-powered decoder — translates raw data into actionable trade theses",
    PHANTOM: "Runs ghost trades through historical data to validate the edge",
}
