"""
Report Generator — prints backtest results, comparisons, and trade logs.
"""

import pandas as pd
from tabulate import tabulate

from backtest.engine import BacktestResult


def print_strategy_report(result: BacktestResult):
    """Print detailed report for a single strategy backtest."""
    print(f"\n{'='*70}")
    print(f"  STRATEGY: {result.strategy_name}")
    print(f"  SYMBOL:   {result.symbol}")
    print(f"{'='*70}")

    metrics = [
        ["Initial Capital", f"Rs.{result.initial_capital:,.0f}"],
        ["Final Capital", f"Rs.{result.final_capital:,.2f}"],
        ["Total P&L", f"Rs.{result.total_pnl:,.2f}"],
        ["Total Return", f"{result.total_return_pct:.2f}%"],
        ["", ""],
        ["Total Trades", result.total_trades],
        ["Winning Trades", result.winning_trades],
        ["Losing Trades", result.losing_trades],
        ["Win Rate", f"{result.win_rate:.1f}%"],
        ["", ""],
        ["Avg Win", f"Rs.{result.avg_win:,.2f}"],
        ["Avg Loss", f"Rs.{result.avg_loss:,.2f}"],
        ["Expectancy/Trade", f"Rs.{result.expectancy:,.2f}"],
        ["Profit Factor", f"{result.profit_factor:.2f}"],
        ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
        ["", ""],
        ["Max Drawdown", f"{result.max_drawdown_pct:.2f}%"],
        ["Max Consecutive Wins", result.max_consecutive_wins],
        ["Max Consecutive Losses", result.max_consecutive_losses],
        ["Avg Trade Duration", f"{result.avg_trade_duration_mins:.0f} mins"],
    ]

    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="simple"))

    # Trade log (last 15 trades)
    if result.trades:
        print(f"\n  Last {min(15, len(result.trades))} Trades:")
        trade_rows = []
        for t in result.trades[-15:]:
            trade_rows.append([
                str(t.entry_time)[:16] if t.entry_time else "",
                t.direction,
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}" if t.exit_price else "",
                f"{t.pnl:+.2f}",
                t.exit_reason,
            ])
        print(tabulate(
            trade_rows,
            headers=["Entry Time", "Dir", "Entry", "Exit", "P&L", "Exit Reason"],
            tablefmt="simple",
        ))


def print_comparison_table(results: list[BacktestResult]):
    """Print side-by-side comparison of all strategy results."""
    print(f"\n{'#'*70}")
    print(f"  STRATEGY COMPARISON — BACKTEST RESULTS")
    print(f"{'#'*70}\n")

    rows = []
    for r in results:
        rows.append([
            r.strategy_name,
            r.symbol,
            r.total_trades,
            f"{r.win_rate:.1f}%",
            f"Rs.{r.total_pnl:,.2f}",
            f"{r.total_return_pct:.2f}%",
            f"{r.profit_factor:.2f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.max_drawdown_pct:.2f}%",
            f"Rs.{r.expectancy:,.2f}",
        ])

    print(tabulate(
        rows,
        headers=[
            "Strategy", "Symbol", "Trades", "Win%",
            "Total P&L", "Return%", "PF", "Sharpe",
            "MaxDD%", "Expectancy",
        ],
        tablefmt="grid",
    ))


def rank_strategies(results: list[BacktestResult]) -> list[BacktestResult]:
    """
    Rank strategies using a composite score:
      - 30% weight: Profit Factor
      - 25% weight: Win Rate
      - 20% weight: Sharpe Ratio
      - 15% weight: Total Return %
      - 10% weight: Inverse Max Drawdown
    Returns sorted list (best first).
    """
    if not results:
        return []

    scored = []
    for r in results:
        # Normalize each metric to 0-100 scale
        pf_score = min(r.profit_factor * 20, 100)  # PF of 5 = 100
        wr_score = r.win_rate  # already 0-100
        sharpe_score = min(max(r.sharpe_ratio * 20, 0), 100)  # Sharpe of 5 = 100
        ret_score = min(max(r.total_return_pct * 2, 0), 100)  # 50% return = 100
        dd_score = max(100 - r.max_drawdown_pct * 5, 0)  # 20% DD = 0

        composite = (
            0.30 * pf_score
            + 0.25 * wr_score
            + 0.20 * sharpe_score
            + 0.15 * ret_score
            + 0.10 * dd_score
        )
        scored.append((composite, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*70}")
    print("  STRATEGY RANKING (Composite Score)")
    print(f"{'='*70}")
    rank_rows = []
    for i, (score, r) in enumerate(scored, 1):
        medal = {1: "[BEST]", 2: "[2nd]", 3: "[3rd]"}.get(i, f"[{i}th]")
        rank_rows.append([
            f"{medal}",
            r.strategy_name,
            r.symbol,
            f"{score:.1f}/100",
            f"{r.win_rate:.1f}%",
            f"{r.profit_factor:.2f}",
            f"Rs.{r.total_pnl:,.2f}",
        ])

    print(tabulate(
        rank_rows,
        headers=["Rank", "Strategy", "Symbol", "Score", "Win%", "PF", "P&L"],
        tablefmt="grid",
    ))

    return [r for _, r in scored]


def print_best_strategy_summary(best: BacktestResult):
    """Print a detailed summary of the best strategy with trading rules."""
    print(f"\n{'*'*70}")
    print(f"  RECOMMENDED STRATEGY: {best.strategy_name}")
    print(f"  Symbol: {best.symbol}")
    print(f"{'*'*70}")
    print(f"""
  Performance Summary:
  ─────────────────────
  Win Rate:       {best.win_rate:.1f}%
  Profit Factor:  {best.profit_factor:.2f}
  Total Return:   {best.total_return_pct:.2f}%
  Max Drawdown:   {best.max_drawdown_pct:.2f}%
  Total Trades:   {best.total_trades}
  Expectancy:     Rs.{best.expectancy:,.2f} per trade

  Risk Management Rules:
  ──────────────────────
  - Risk per trade: 1% of capital
  - Max trades per day: 6
  - No new trades after 14:45
  - Force exit all positions by 15:15
  - SL: 1.5x ATR from entry
  - Target: 2.0x ATR from entry (1:1.33 RR minimum)
  - Trailing SL: 1.0x ATR (locks in profits)
""")
