"""
Variant 1: EMA 9/20 Crossover + SuperTrend Confirmation (CLASSIC)
─────────────────────────────────────────────────────────────────
The safest variant. Only enters when BOTH EMA crossover and SuperTrend agree.

LOGIC:
  BUY when:
    - EMA 9 crosses above EMA 20 (bullish crossover)
    - SuperTrend is bullish (direction = 1)
    - Close is above EMA 20 (price respecting the fast trend)
    - Candle closes above SuperTrend level

  SELL when:
    - EMA 9 crosses below EMA 20 (bearish crossover)
    - SuperTrend is bearish (direction = -1)
    - Close is below EMA 20
    - Candle closes below SuperTrend level

  SL: SuperTrend level (natural support/resistance)
  Target: 2x ATR from entry (minimum 1:1.5 RR)

  WHY THIS WORKS:
    - Dual confirmation eliminates most false crossover signals
    - SuperTrend acts as a macro trend filter
    - Fewer trades but higher win rate
    - Best in trending markets, avoids choppy range-bound days
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyEma920StClassic(BaseStrategy):

    @property
    def name(self) -> str:
        return "V1: EMA9/20 Cross + ST Confirm (Classic)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Signal"] = "HOLD"
        df["SL"] = np.nan
        df["Target"] = np.nan
        df["Reason"] = ""

        time_ok = self._time_filter(df)

        for i in range(2, len(df)):
            if not time_ok.iloc[i]:
                continue

            close = df["Close"].iloc[i]
            ema9 = df["EMA_9"].iloc[i]
            ema20 = df["EMA_20"].iloc[i]
            prev_ema9 = df["EMA_9"].iloc[i - 1]
            prev_ema20 = df["EMA_20"].iloc[i - 1]
            st = df["SuperTrend"].iloc[i]
            st_dir = df["ST_Direction"].iloc[i]
            atr = df["ATR"].iloc[i]

            if pd.isna(ema20) or pd.isna(st) or pd.isna(atr):
                continue

            # EMA crossovers
            ema_cross_up = (ema9 > ema20) and (prev_ema9 <= prev_ema20)
            ema_cross_down = (ema9 < ema20) and (prev_ema9 >= prev_ema20)

            # ── BUY: Crossover UP + SuperTrend Bullish ──
            if ema_cross_up and st_dir == 1 and close > ema20 and close > st:
                sl = max(st, close - config.SL_ATR_MULTIPLIER * atr)
                target = close + config.TARGET_ATR_MULTIPLIER * atr

                # Ensure minimum 1:1.5 RR
                risk = close - sl
                if risk > 0 and target < close + 1.5 * risk:
                    target = close + 1.5 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: EMA9({ema9:.0f}) crossed above EMA20({ema20:.0f}), "
                    f"ST bullish, Close({close:.0f})>ST({st:.0f})"
                )
                continue

            # ── SELL: Crossover DOWN + SuperTrend Bearish ──
            if ema_cross_down and st_dir == -1 and close < ema20 and close < st:
                sl = min(st, close + config.SL_ATR_MULTIPLIER * atr)
                target = close - config.TARGET_ATR_MULTIPLIER * atr

                risk = sl - close
                if risk > 0 and target > close - 1.5 * risk:
                    target = close - 1.5 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: EMA9({ema9:.0f}) crossed below EMA20({ema20:.0f}), "
                    f"ST bearish, Close({close:.0f})<ST({st:.0f})"
                )

        return df
