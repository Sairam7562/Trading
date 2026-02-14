#!/usr/bin/env python3
"""
Intraday Trading Strategy Backtester
─────────────────────────────────────
Runs all strategies on NIFTY and BANKNIFTY 5-min data,
backtests them, compares results, optimizes parameters,
and identifies the best variant.

Usage:
    python run_backtest.py                  # Try live data, fallback to synthetic
    python run_backtest.py --synthetic      # Force synthetic data
    python run_backtest.py --symbol NIFTY   # Single symbol only
    python run_backtest.py --optimize       # Run parameter optimization
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetcher import load_data
from utils.indicators import compute_all_indicators
from strategies.strategy_ema_cpr_rsi import StrategyEmaCprRsi
from strategies.strategy_ema_vwap_supertrend import StrategyEmaVwapSupertrend
from strategies.strategy_cpr_vwap_ema_crossover import StrategyCprVwapEmaCrossover
from backtest.engine import run_backtest
from backtest.optimizer import optimize_atr_params, print_optimization_results
from reports.report import (
    print_strategy_report,
    print_comparison_table,
    rank_strategies,
    print_best_strategy_summary,
)


def get_all_strategies():
    """Return all strategy instances to backtest."""
    return [
        StrategyEmaCprRsi(),
        StrategyEmaVwapSupertrend(),
        StrategyCprVwapEmaCrossover(cpr_width_threshold=0.3),  # Conservative
        StrategyCprVwapEmaCrossover(cpr_width_threshold=0.5),  # Aggressive
    ]


def run_optimization(symbols, use_synthetic):
    """Run parameter optimization for each strategy on each symbol."""
    print(f"\n{'#'*70}")
    print("  PARAMETER OPTIMIZATION")
    print(f"{'#'*70}")

    strategy_configs = [
        (StrategyEmaCprRsi, {}, "EMA50 + CPR + RSI"),
        (StrategyEmaVwapSupertrend, {}, "EMA + VWAP + SuperTrend"),
        (StrategyCprVwapEmaCrossover, {"cpr_width_threshold": 0.3}, "CPR+VWAP+EMA (0.3%)"),
        (StrategyCprVwapEmaCrossover, {"cpr_width_threshold": 0.5}, "CPR+VWAP+EMA (0.5%)"),
    ]

    best_overall = None
    best_overall_score = -1
    best_overall_params = None
    best_overall_name = ""
    best_overall_symbol = ""

    for symbol_key in symbols:
        df = load_data(symbol_key, use_synthetic=use_synthetic)
        df_ind = compute_all_indicators(df)

        for cls, kwargs, name in strategy_configs:
            print(f"\n  Optimizing: {name} on {symbol_key}...")
            scored = optimize_atr_params(df_ind, cls, symbol_key, kwargs if kwargs else None)
            print_optimization_results(scored, top_n=3)

            if scored and scored[0][0] > best_overall_score:
                best_overall_score = scored[0][0]
                best_overall_params = scored[0][1]
                best_overall = scored[0][2]
                best_overall_name = name
                best_overall_symbol = symbol_key

    if best_overall:
        print(f"\n{'*'*70}")
        print(f"  BEST OPTIMIZED CONFIGURATION")
        print(f"{'*'*70}")
        print(f"  Strategy:  {best_overall_name}")
        print(f"  Symbol:    {best_overall_symbol}")
        print(f"  Score:     {best_overall_score:.1f}/100")
        print(f"  Parameters:")
        for k, v in best_overall_params.items():
            print(f"    {k}: {v}")
        print_strategy_report(best_overall)


def main():
    parser = argparse.ArgumentParser(description="Intraday Strategy Backtester")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--symbol", type=str, default=None, help="Run for single symbol (NIFTY or BANKNIFTY)")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(config.SYMBOLS.keys())
    strategies = get_all_strategies()

    print("=" * 70)
    print("  INTRADAY TRADING STRATEGY BACKTESTER")
    print("  Timeframe: 5-minute | Market: NSE (India)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Capital: Rs.{config.INITIAL_CAPITAL:,.0f}")
    print("=" * 70)

    all_results = []

    for symbol_key in symbols:
        print(f"\n{'─'*70}")
        print(f"  Processing {symbol_key}...")
        print(f"{'─'*70}")

        # 1. Load data
        df = load_data(symbol_key, use_synthetic=args.synthetic)
        print(f"  [DATA] Loaded {len(df)} candles")

        # 2. Compute indicators
        print(f"  [INDICATORS] Computing EMA, CPR, VWAP, RSI, ATR, SuperTrend...")
        df_with_indicators = compute_all_indicators(df)
        print(f"  [INDICATORS] Done. Columns: {list(df_with_indicators.columns)}")

        # 3. Run each strategy
        for strategy in strategies:
            print(f"\n  [STRATEGY] Running: {strategy.name}...")

            # Generate signals
            df_signals = strategy.generate_signals(df_with_indicators)

            # Count signals
            buy_count = (df_signals["Signal"] == "BUY").sum()
            sell_count = (df_signals["Signal"] == "SELL").sum()
            print(f"  [SIGNALS] BUY={buy_count}, SELL={sell_count}")

            # Run backtest
            result = run_backtest(df_signals, strategy.name, symbol_key)
            all_results.append(result)

            # Print individual report
            print_strategy_report(result)

    # ── Comparison & Ranking ──
    if all_results:
        print_comparison_table(all_results)
        ranked = rank_strategies(all_results)

        if ranked:
            print_best_strategy_summary(ranked[0])

    # ── Optimization ──
    if args.optimize:
        run_optimization(symbols, args.synthetic)

    print(f"\n{'='*70}")
    print("  Backtest complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
