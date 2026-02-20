"""
Report Generator — prints backtest results, comparisons, and trade logs.
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import defaultdict

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


def print_detailed_trade_analysis(result: BacktestResult):
    """
    Print comprehensive trade-by-trade analysis with:
    - Full trade log (entry, exit, SL, target, P&L, running capital)
    - Exit reason breakdown
    - Daily P&L summary
    - Biggest wins/losses
    - P&L distribution
    - Equity curve data
    - Clear exit rules
    """
    if not result.trades:
        print("\n  No trades to analyze.")
        return

    trades = result.trades
    capital = result.initial_capital

    # ── HEADER ──
    print(f"\n{'='*90}")
    print(f"  DETAILED TRADE ANALYSIS: {result.strategy_name}")
    print(f"  Symbol: {result.symbol} | Capital: Rs.{capital:,.0f}")
    print(f"{'='*90}")

    # ── EXIT RULES ──
    print(f"\n  EXIT RULES (How & When Trades Close):")
    print(f"  {'─'*50}")
    print(f"  1. TARGET HIT  — Price reaches 2.0x ATR from entry (profit booking)")
    print(f"  2. SL HIT      — Price hits stop-loss (SuperTrend level or 1.5x ATR)")
    print(f"  3. TRAILING SL — SL moves up with price (1.0x ATR trail), locks profit")
    print(f"  4. TIME EXIT   — All positions squared off at 15:15 (end of day)")
    print(f"  5. OPPOSITE SIGNAL — New signal in opposite direction closes current trade")

    # ── COMPLETE TRADE LOG ──
    print(f"\n  COMPLETE TRADE LOG ({len(trades)} trades):")
    print(f"  {'─'*85}")

    running_capital = capital
    trade_rows = []
    equity_points = [capital]

    for idx, t in enumerate(trades, 1):
        running_capital += t.pnl
        equity_points.append(running_capital)

        entry_str = str(t.entry_time)[:16] if t.entry_time else ""
        exit_str = str(t.exit_time)[:16] if t.exit_time else ""
        duration = ""
        if t.entry_time and t.exit_time:
            dur_mins = int((t.exit_time - t.entry_time).total_seconds() / 60)
            duration = f"{dur_mins}m"

        win_loss = "WIN" if t.pnl > 0 else "LOSS"
        pnl_str = f"{t.pnl:+.2f}"

        trade_rows.append([
            idx,
            entry_str,
            exit_str,
            t.direction,
            f"{t.entry_price:.2f}",
            f"{t.sl:.2f}" if t.sl else "",
            f"{t.target:.2f}" if t.target else "",
            f"{t.exit_price:.2f}" if t.exit_price else "",
            pnl_str,
            win_loss,
            t.exit_reason,
            duration,
            f"{running_capital:,.0f}",
        ])

    print(tabulate(
        trade_rows,
        headers=[
            "#", "Entry Time", "Exit Time", "Dir", "Entry",
            "SL", "Target", "Exit", "P&L (pts)", "W/L",
            "Exit Reason", "Duration", "Capital",
        ],
        tablefmt="simple",
        numalign="right",
    ))

    # ── EXIT REASON BREAKDOWN ──
    print(f"\n  EXIT REASON BREAKDOWN:")
    print(f"  {'─'*50}")

    exit_reasons = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "wins": 0, "losses": 0})
    for t in trades:
        reason = t.exit_reason or "Unknown"
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["total_pnl"] += t.pnl
        if t.pnl > 0:
            exit_reasons[reason]["wins"] += 1
        else:
            exit_reasons[reason]["losses"] += 1

    reason_rows = []
    for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
        wr = (data["wins"] / data["count"] * 100) if data["count"] > 0 else 0
        reason_rows.append([
            reason,
            data["count"],
            f"{data['count'] / len(trades) * 100:.1f}%",
            data["wins"],
            data["losses"],
            f"{wr:.0f}%",
            f"{data['total_pnl']:+,.2f}",
        ])

    print(tabulate(
        reason_rows,
        headers=["Exit Reason", "Count", "% of Total", "Wins", "Losses", "Win%", "Total P&L"],
        tablefmt="simple",
    ))

    # ── DAILY P&L BREAKDOWN ──
    print(f"\n  DAILY P&L BREAKDOWN:")
    print(f"  {'─'*50}")

    daily_pnl = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    for t in trades:
        if t.entry_time:
            day = str(t.entry_time.date()) if hasattr(t.entry_time, 'date') else str(t.entry_time)[:10]
            daily_pnl[day]["trades"] += 1
            daily_pnl[day]["pnl"] += t.pnl
            if t.pnl > 0:
                daily_pnl[day]["wins"] += 1
            else:
                daily_pnl[day]["losses"] += 1

    daily_rows = []
    cumulative_pnl = 0.0
    for day in sorted(daily_pnl.keys()):
        data = daily_pnl[day]
        cumulative_pnl += data["pnl"]
        day_result = "GREEN" if data["pnl"] > 0 else ("RED" if data["pnl"] < 0 else "FLAT")
        daily_rows.append([
            day,
            data["trades"],
            data["wins"],
            data["losses"],
            f"{data['pnl']:+,.2f}",
            day_result,
            f"{cumulative_pnl:+,.2f}",
        ])

    print(tabulate(
        daily_rows,
        headers=["Date", "Trades", "Wins", "Losses", "Day P&L", "Result", "Cumulative P&L"],
        tablefmt="simple",
    ))

    green_days = sum(1 for d in daily_pnl.values() if d["pnl"] > 0)
    red_days = sum(1 for d in daily_pnl.values() if d["pnl"] < 0)
    flat_days = sum(1 for d in daily_pnl.values() if d["pnl"] == 0)
    total_days = len(daily_pnl)
    print(f"\n  Trading Days: {total_days} | Green: {green_days} | Red: {red_days} | Flat: {flat_days}")
    if total_days > 0:
        print(f"  Green Day %: {green_days / total_days * 100:.1f}%")

    # ── BIGGEST WINS & LOSSES ──
    print(f"\n  TOP 5 WINS:")
    print(f"  {'─'*50}")
    sorted_wins = sorted([t for t in trades if t.pnl > 0], key=lambda x: x.pnl, reverse=True)
    win_rows = []
    for t in sorted_wins[:5]:
        win_rows.append([
            str(t.entry_time)[:16],
            t.direction,
            f"{t.entry_price:.2f}",
            f"{t.exit_price:.2f}",
            f"{t.pnl:+,.2f}",
            f"{t.pnl_pct:+.2f}%",
            t.exit_reason,
        ])
    if win_rows:
        print(tabulate(win_rows, headers=["Time", "Dir", "Entry", "Exit", "P&L", "P&L%", "Exit"], tablefmt="simple"))
    else:
        print("  No winning trades")

    print(f"\n  TOP 5 LOSSES:")
    print(f"  {'─'*50}")
    sorted_losses = sorted([t for t in trades if t.pnl <= 0], key=lambda x: x.pnl)
    loss_rows = []
    for t in sorted_losses[:5]:
        loss_rows.append([
            str(t.entry_time)[:16],
            t.direction,
            f"{t.entry_price:.2f}",
            f"{t.exit_price:.2f}",
            f"{t.pnl:+,.2f}",
            f"{t.pnl_pct:+.2f}%",
            t.exit_reason,
        ])
    if loss_rows:
        print(tabulate(loss_rows, headers=["Time", "Dir", "Entry", "Exit", "P&L", "P&L%", "Exit"], tablefmt="simple"))
    else:
        print("  No losing trades")

    # ── P&L DISTRIBUTION ──
    print(f"\n  P&L DISTRIBUTION:")
    print(f"  {'─'*50}")
    pnls = [t.pnl for t in trades]
    pnl_arr = np.array(pnls)

    dist_metrics = [
        ["Biggest Win", f"Rs.{max(pnls):+,.2f}" if pnls else "N/A"],
        ["Biggest Loss", f"Rs.{min(pnls):+,.2f}" if pnls else "N/A"],
        ["Average P&L", f"Rs.{np.mean(pnl_arr):+,.2f}"],
        ["Median P&L", f"Rs.{np.median(pnl_arr):+,.2f}"],
        ["Std Dev", f"Rs.{np.std(pnl_arr):,.2f}"],
        ["", ""],
        ["Gross Profit", f"Rs.{sum(p for p in pnls if p > 0):+,.2f}"],
        ["Gross Loss", f"Rs.{sum(p for p in pnls if p <= 0):+,.2f}"],
        ["Net P&L", f"Rs.{sum(pnls):+,.2f}"],
    ]
    print(tabulate(dist_metrics, headers=["Metric", "Value"], tablefmt="simple"))

    # ── EQUITY CURVE (text-based) ──
    print(f"\n  EQUITY CURVE:")
    print(f"  {'─'*50}")
    eq = np.array(equity_points)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100

    eq_rows = [
        ["Starting Capital", f"Rs.{equity_points[0]:,.0f}"],
        ["Final Capital", f"Rs.{equity_points[-1]:,.0f}"],
        ["Peak Capital", f"Rs.{max(equity_points):,.0f}"],
        ["Lowest Capital", f"Rs.{min(equity_points):,.0f}"],
        ["Max Drawdown", f"{dd.max():.2f}%"],
        ["Max Drawdown (Rs.)", f"Rs.{(peak - eq).max():,.2f}"],
    ]
    print(tabulate(eq_rows, headers=["Metric", "Value"], tablefmt="simple"))

    # Mini ASCII equity curve
    _print_ascii_equity(equity_points)

    # ── LONG vs SHORT ANALYSIS ──
    print(f"\n  LONG vs SHORT BREAKDOWN:")
    print(f"  {'─'*50}")

    for direction in ["BUY", "SELL"]:
        dir_trades = [t for t in trades if t.direction == direction]
        if not dir_trades:
            continue
        dir_pnls = [t.pnl for t in dir_trades]
        dir_wins = sum(1 for p in dir_pnls if p > 0)
        label = "LONG" if direction == "BUY" else "SHORT"
        print(f"  {label}:  Trades={len(dir_trades)}  Wins={dir_wins}  "
              f"Win%={dir_wins / len(dir_trades) * 100:.0f}%  "
              f"P&L=Rs.{sum(dir_pnls):+,.2f}  "
              f"Avg=Rs.{np.mean(dir_pnls):+,.2f}")

    print(f"\n{'='*90}")


def _print_ascii_equity(equity_points, width=60, height=15):
    """Print a simple ASCII chart of the equity curve."""
    if len(equity_points) < 2:
        return

    eq = np.array(equity_points, dtype=float)
    n = len(eq)

    # Downsample to fit width
    if n > width:
        indices = np.linspace(0, n - 1, width).astype(int)
        eq_sampled = eq[indices]
    else:
        eq_sampled = eq
        width = len(eq_sampled)

    min_val = eq_sampled.min()
    max_val = eq_sampled.max()
    val_range = max_val - min_val

    if val_range == 0:
        return

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    for col in range(width):
        row = int((eq_sampled[col] - min_val) / val_range * (height - 1))
        row = height - 1 - row  # flip for display
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    # Print
    print(f"\n  Rs.{max_val:,.0f} |", end="")
    for row in range(height):
        prefix = "            |" if row > 0 else ""
        if row > 0:
            print(f"\n  {prefix}", end="")
        print("".join(grid[row]), end="|")

    print(f"\n  Rs.{min_val:,.0f} |{'─' * width}|")
    print(f"  {'':>13}Trade #1{' ' * (width - 16)}Trade #{len(equity_points) - 1}")
