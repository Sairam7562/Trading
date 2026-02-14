"""
Parameter Optimizer
───────────────────
Tests different parameter combinations for each strategy
to find the optimal settings. Uses grid search over key parameters.
"""

import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backtest.engine import run_backtest, BacktestResult


def optimize_atr_params(df_with_indicators, strategy_class, symbol, strategy_kwargs=None):
    """
    Grid search over SL/Target ATR multipliers and trailing SL.
    Returns list of (params, result) tuples sorted by composite score.
    """
    sl_multipliers = [1.0, 1.5, 2.0]
    target_multipliers = [1.5, 2.0, 3.0]
    trailing_multipliers = [0.75, 1.0, 1.5]

    results = []
    total = len(sl_multipliers) * len(target_multipliers) * len(trailing_multipliers)
    tested = 0

    for sl_m, tgt_m, trail_m in itertools.product(sl_multipliers, target_multipliers, trailing_multipliers):
        # Skip if target < SL (bad RR)
        if tgt_m <= sl_m:
            continue

        tested += 1

        # Temporarily override config
        orig_sl = config.SL_ATR_MULTIPLIER
        orig_tgt = config.TARGET_ATR_MULTIPLIER
        orig_trail = config.TRAILING_SL_ATR_MULTIPLIER

        config.SL_ATR_MULTIPLIER = sl_m
        config.TARGET_ATR_MULTIPLIER = tgt_m
        config.TRAILING_SL_ATR_MULTIPLIER = trail_m

        try:
            kwargs = strategy_kwargs or {}
            strategy = strategy_class(**kwargs)
            df_signals = strategy.generate_signals(df_with_indicators)
            result = run_backtest(df_signals, strategy.name, symbol)

            params = {
                "SL_ATR": sl_m,
                "Target_ATR": tgt_m,
                "Trail_ATR": trail_m,
            }
            if strategy_kwargs:
                params.update(strategy_kwargs)

            results.append((params, result))
        finally:
            # Restore config
            config.SL_ATR_MULTIPLIER = orig_sl
            config.TARGET_ATR_MULTIPLIER = orig_tgt
            config.TRAILING_SL_ATR_MULTIPLIER = orig_trail

    # Score and sort
    scored = []
    for params, result in results:
        if result.total_trades == 0:
            score = 0
        else:
            pf_score = min(result.profit_factor * 20, 100)
            wr_score = result.win_rate
            sharpe_score = min(max(result.sharpe_ratio * 20, 0), 100)
            ret_score = min(max(result.total_return_pct * 2, 0), 100)
            dd_score = max(100 - result.max_drawdown_pct * 5, 0)

            score = (
                0.30 * pf_score
                + 0.25 * wr_score
                + 0.20 * sharpe_score
                + 0.15 * ret_score
                + 0.10 * dd_score
            )
        scored.append((score, params, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def print_optimization_results(scored, top_n=5):
    """Print top N parameter combinations."""
    from tabulate import tabulate

    print(f"\n  Top {min(top_n, len(scored))} Parameter Combinations:")
    rows = []
    for i, (score, params, result) in enumerate(scored[:top_n], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        rows.append([
            i,
            param_str,
            result.total_trades,
            f"{result.win_rate:.1f}%",
            f"Rs.{result.total_pnl:,.2f}",
            f"{result.profit_factor:.2f}",
            f"{result.max_drawdown_pct:.2f}%",
            f"{score:.1f}",
        ])

    print(tabulate(
        rows,
        headers=["#", "Parameters", "Trades", "Win%", "P&L", "PF", "MaxDD%", "Score"],
        tablefmt="grid",
    ))
