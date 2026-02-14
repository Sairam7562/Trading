"""
Backtesting Engine
──────────────────
Simulates trades from strategy signals on historical data.
Handles: position sizing, SL/Target exits, trailing SL, time-based exit,
         max trades per day, brokerage, slippage.
"""

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class Trade:
    entry_time: dt.datetime
    exit_time: dt.datetime = None
    direction: str = ""        # BUY or SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    target: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    reason: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    trades: list = field(default_factory=list)
    initial_capital: float = config.INITIAL_CAPITAL
    final_capital: float = 0.0

    # Computed metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_mins: float = 0.0
    expectancy: float = 0.0

    def compute_metrics(self):
        if not self.trades:
            return

        pnls = [t.pnl for t in self.trades]
        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for p in pnls if p > 0)
        self.losing_trades = sum(1 for p in pnls if p <= 0)
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades else 0

        self.total_pnl = sum(pnls)
        self.final_capital = self.initial_capital + self.total_pnl
        self.total_return_pct = (self.total_pnl / self.initial_capital) * 100

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        if self.total_trades > 0:
            self.expectancy = self.total_pnl / self.total_trades

        # Max drawdown
        equity_curve = [self.initial_capital]
        for p in pnls:
            equity_curve.append(equity_curve[-1] + p)
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100
        self.max_drawdown_pct = drawdown.max()

        # Sharpe ratio (daily returns approximation)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_capital
            self.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Consecutive wins/losses
        self.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        self.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        # Average trade duration
        durations = []
        for t in self.trades:
            if t.exit_time and t.entry_time:
                dur = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(dur)
        self.avg_trade_duration_mins = np.mean(durations) if durations else 0

    @staticmethod
    def _max_consecutive(pnls, positive=True):
        max_streak = 0
        current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p <= 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak


def run_backtest(df: pd.DataFrame, strategy_name: str, symbol: str) -> BacktestResult:
    """
    Run backtest on a DataFrame that already has Signal, SL, Target columns.
    Simulates actual trade execution with proper entry/exit logic.
    """
    result = BacktestResult(strategy_name=strategy_name, symbol=symbol)
    capital = config.INITIAL_CAPITAL

    # Group by date for max-trades-per-day enforcement
    df["_date"] = df.index.date

    position = None  # Current open position (Trade object)
    trades_today = 0
    current_date = None

    force_exit_time = dt.time(
        int(config.FORCE_EXIT_TIME.split(":")[0]),
        int(config.FORCE_EXIT_TIME.split(":")[1]),
    )

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        row_date = ts.date() if hasattr(ts, 'date') else ts

        # Reset daily trade counter
        if row_date != current_date:
            current_date = row_date
            trades_today = 0

        # ── Check for exit on open position ──
        if position is not None:
            exit_price = None
            exit_reason = None

            if position.direction == "BUY":
                # SL hit
                if row["Low"] <= position.sl:
                    exit_price = position.sl - config.SLIPPAGE_POINTS
                    exit_reason = "SL Hit"
                # Target hit
                elif row["High"] >= position.target:
                    exit_price = position.target - config.SLIPPAGE_POINTS
                    exit_reason = "Target Hit"
                # Time-based exit
                elif ts.time() >= force_exit_time:
                    exit_price = row["Close"]
                    exit_reason = "Time Exit"

            elif position.direction == "SELL":
                # SL hit
                if row["High"] >= position.sl:
                    exit_price = position.sl + config.SLIPPAGE_POINTS
                    exit_reason = "SL Hit"
                # Target hit
                elif row["Low"] <= position.target:
                    exit_price = position.target + config.SLIPPAGE_POINTS
                    exit_reason = "Target Hit"
                # Time-based exit
                elif ts.time() >= force_exit_time:
                    exit_price = row["Close"]
                    exit_reason = "Time Exit"

            # Trailing SL update (if position still open)
            if exit_price is None and position is not None:
                atr = row.get("ATR", None)
                if atr and not pd.isna(atr):
                    if position.direction == "BUY":
                        new_sl = row["Close"] - config.TRAILING_SL_ATR_MULTIPLIER * atr
                        if new_sl > position.sl:
                            position.sl = new_sl
                    elif position.direction == "SELL":
                        new_sl = row["Close"] + config.TRAILING_SL_ATR_MULTIPLIER * atr
                        if new_sl < position.sl:
                            position.sl = new_sl

            if exit_price is not None:
                position.exit_time = ts
                position.exit_price = exit_price
                position.exit_reason = exit_reason

                # Calculate PnL
                if position.direction == "BUY":
                    position.pnl = exit_price - position.entry_price
                else:
                    position.pnl = position.entry_price - exit_price

                # Subtract brokerage (entry + exit)
                position.pnl -= 2 * config.BROKERAGE_PER_TRADE / max(1, abs(position.entry_price))
                position.pnl_pct = (position.pnl / position.entry_price) * 100

                capital += position.pnl
                result.trades.append(position)
                position = None
                continue

        # ── Check for new entry ──
        if position is None and trades_today < config.MAX_TRADES_PER_DAY:
            signal = row.get("Signal", "HOLD")
            if signal in ("BUY", "SELL"):
                entry_price = row["Close"]
                if signal == "BUY":
                    entry_price += config.SLIPPAGE_POINTS
                else:
                    entry_price -= config.SLIPPAGE_POINTS

                position = Trade(
                    entry_time=ts,
                    direction=signal,
                    entry_price=entry_price,
                    sl=row["SL"],
                    target=row["Target"],
                    reason=row.get("Reason", ""),
                )
                trades_today += 1

    # Close any remaining position at last close
    if position is not None:
        last_row = df.iloc[-1]
        position.exit_time = df.index[-1]
        position.exit_price = last_row["Close"]
        position.exit_reason = "End of Data"

        if position.direction == "BUY":
            position.pnl = position.exit_price - position.entry_price
        else:
            position.pnl = position.entry_price - position.exit_price

        position.pnl -= 2 * config.BROKERAGE_PER_TRADE / max(1, abs(position.entry_price))
        position.pnl_pct = (position.pnl / position.entry_price) * 100
        capital += position.pnl
        result.trades.append(position)

    df.drop(columns=["_date"], inplace=True, errors="ignore")

    result.compute_metrics()
    return result
