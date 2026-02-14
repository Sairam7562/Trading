# Intraday Trading Strategy Backtester

Backtesting framework for **NIFTY 50** and **BANKNIFTY** intraday strategies on 5-minute charts.

## Strategies

### Strategy 1: EMA 50 + CPR + RSI (Best Performer)
- **BUY**: Price > EMA 50, above Top CPR, RSI recovering from oversold, above VWAP
- **SELL**: Price < EMA 50, below Bottom CPR, RSI falling from overbought, below VWAP
- **Best on**: BANKNIFTY with SL=1.0x ATR, Target=3.0x ATR, Trail=0.75x ATR

### Strategy 2: EMA + VWAP + SuperTrend
- **BUY**: SuperTrend flips bullish or pullback bounce, price > EMA 21, above VWAP
- **SELL**: SuperTrend flips bearish or pullback rejection, price < EMA 21, below VWAP
- **Best on**: Trending market days with clean directional moves

### Strategy 3: CPR + VWAP + EMA Crossover
- Uses narrow CPR to detect trending days
- **BUY**: EMA 9/21 crossover + CPR breakout above TC + VWAP confirmation
- **SELL**: EMA 9/21 death cross + CPR breakdown below BC + VWAP rejection
- Two variants: Conservative (CPR < 0.3%) and Aggressive (CPR < 0.5%)

## Indicators Used
| Indicator | Purpose |
|-----------|---------|
| EMA 9/21/50 | Trend direction and crossover signals |
| CPR (Central Pivot Range) | Support/resistance zones from previous day |
| VWAP | Institutional buying/selling bias |
| RSI (14) | Momentum and overbought/oversold detection |
| SuperTrend (10, 3.0) | Trend following with dynamic SL |
| ATR (14) | Volatility-based SL/Target calculation |

## Optimized Parameters (Backtest Results)

| Parameter | Value |
|-----------|-------|
| SL | 1.0x ATR |
| Target | 3.0x ATR (1:3 RR) |
| Trailing SL | 0.75x ATR |
| Max trades/day | 6 |
| No new trades after | 14:45 |
| Force exit | 15:15 |

## Usage

```bash
pip install -r requirements.txt

# Run with synthetic data
python run_backtest.py --synthetic

# Run with live Yahoo Finance data
python run_backtest.py

# Single symbol
python run_backtest.py --symbol BANKNIFTY

# With parameter optimization
python run_backtest.py --synthetic --optimize
```

## Project Structure

```
Trading/
├── config.py                  # All configurable parameters
├── run_backtest.py            # Main entry point
├── requirements.txt
├── data/
│   └── fetcher.py             # Data loading (yfinance + synthetic)
├── utils/
│   └── indicators.py          # EMA, CPR, VWAP, RSI, SuperTrend, ATR
├── strategies/
│   ├── base.py                # Base strategy interface
│   ├── strategy_ema_cpr_rsi.py
│   ├── strategy_ema_vwap_supertrend.py
│   └── strategy_cpr_vwap_ema_crossover.py
├── backtest/
│   ├── engine.py              # Trade simulation engine
│   └── optimizer.py           # Parameter grid search
└── reports/
    └── report.py              # Results display and ranking
```
