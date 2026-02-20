"""
Configuration for Intraday Trading Strategy Backtester
Supports: NIFTY 50 and BANKNIFTY on 5-minute charts
"""

# ─── Symbols ───────────────────────────────────────────────
# Yahoo Finance tickers for Indian indices
SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# ─── Timeframe ─────────────────────────────────────────────
INTERVAL = "5m"                # 5-minute candles
LOOKBACK_DAYS = 59             # yfinance max for 5m data is ~60 days
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# ─── Capital & Risk ───────────────────────────────────────
INITIAL_CAPITAL = 100_000      # INR
RISK_PER_TRADE_PCT = 1.0       # risk 1% of capital per trade
MAX_TRADES_PER_DAY = 6         # avoid overtrading
BROKERAGE_PER_TRADE = 40       # flat per-order (Zerodha-like)
SLIPPAGE_POINTS = 2            # assumed slippage in index points

# ─── Indicator Defaults ───────────────────────────────────
EMA_FAST = 9
EMA_20 = 20
EMA_MID = 21
EMA_SLOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
ATR_PERIOD = 14

# ─── Trade Management ─────────────────────────────────────
SL_ATR_MULTIPLIER = 1.5        # stop-loss = entry +/- ATR * multiplier
TARGET_ATR_MULTIPLIER = 2.0    # target  = entry +/- ATR * multiplier
TRAILING_SL_ATR_MULTIPLIER = 1.0  # trailing SL tightens as trade moves

# ─── Session Cutoff ───────────────────────────────────────
NO_NEW_TRADE_AFTER = "14:45"   # no fresh entries after this time
FORCE_EXIT_TIME = "15:15"      # square-off all open positions
