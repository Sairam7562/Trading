"""
Variant 2: EMA 9/20 Crossover Primary + SuperTrend Risk Adjuster
─────────────────────────────────────────────────────────────────
Trades on EVERY EMA crossover, but uses SuperTrend to size risk.

LOGIC:
  BUY when:
    - EMA 9 crosses above EMA 20
    - Close is above EMA 20
    If SuperTrend is ALSO bullish → tight SL (1x ATR), wide target (2.5x ATR)
    If SuperTrend is bearish → wider SL (2x ATR), conservative target (1.5x ATR)

  SELL when:
    - EMA 9 crosses below EMA 20
    - Close is below EMA 20
    Same risk logic mirrored.

  WHY THIS WORKS:
    - Catches ALL crossover moves (more trades than V1)
    - SuperTrend disagreement = smaller position / wider stop
    - Lets you participate even in early reversals before ST flips
    - Good balance of trade frequency and risk management
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyEma920StRisk(BaseStrategy):

    @property
    def name(self) -> str:
        return "V2: EMA9/20 Cross + ST Risk Adjust"

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

            if pd.isna(ema20) or pd.isna(st) or pd.isna(atr) or atr <= 0:
                continue

            ema_cross_up = (ema9 > ema20) and (prev_ema9 <= prev_ema20)
            ema_cross_down = (ema9 < ema20) and (prev_ema9 >= prev_ema20)

            # ── BUY on EMA cross up ──
            if ema_cross_up and close > ema20:
                if st_dir == 1:
                    # ST agrees — aggressive: tight SL, wide target
                    sl = close - 1.0 * atr
                    target = close + 2.5 * atr
                    conf = "ST-aligned"
                else:
                    # ST disagrees — conservative: wider SL, modest target
                    sl = close - 2.0 * atr
                    target = close + 1.5 * atr
                    conf = "ST-contrary"

                risk = close - sl
                if risk > 0 and (target - close) < 1.2 * risk:
                    target = close + 1.2 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: EMA9({ema9:.0f})>EMA20({ema20:.0f}) xover, "
                    f"{conf}, ATR={atr:.1f}"
                )
                continue

            # ── SELL on EMA cross down ──
            if ema_cross_down and close < ema20:
                if st_dir == -1:
                    sl = close + 1.0 * atr
                    target = close - 2.5 * atr
                    conf = "ST-aligned"
                else:
                    sl = close + 2.0 * atr
                    target = close - 1.5 * atr
                    conf = "ST-contrary"

                risk = sl - close
                if risk > 0 and (close - target) < 1.2 * risk:
                    target = close - 1.2 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: EMA9({ema9:.0f})<EMA20({ema20:.0f}) xover, "
                    f"{conf}, ATR={atr:.1f}"
                )

        return df
