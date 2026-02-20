"""
Variant 3: SuperTrend Flip + EMA 9/20 Alignment (Reversal Catcher)
──────────────────────────────────────────────────────────────────
Enters on SuperTrend direction change, confirmed by EMA alignment.

LOGIC:
  BUY when:
    - SuperTrend flips from bearish to bullish (direction -1 → 1)
    - EMA 9 > EMA 20 (EMAs already aligned bullish)
    - Close is above both EMA 9 and EMA 20

  SELL when:
    - SuperTrend flips from bullish to bearish (direction 1 → -1)
    - EMA 9 < EMA 20 (EMAs already aligned bearish)
    - Close is below both EMA 9 and EMA 20

  SL: Previous swing via SuperTrend level + buffer
  Target: 2x ATR (aggressive because catching reversals early)

  WHY THIS WORKS:
    - Catches trend reversals at the earliest confirmation point
    - EMA alignment ensures you're not catching a false ST flip
    - SuperTrend flip is a strong momentum signal
    - Best for volatile days with clear trend changes
    - Naturally avoids sideways markets (ST flips are rare in ranges)
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyEma920StReversal(BaseStrategy):

    @property
    def name(self) -> str:
        return "V3: ST Flip + EMA9/20 Align (Reversal)"

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
            st = df["SuperTrend"].iloc[i]
            st_dir = df["ST_Direction"].iloc[i]
            prev_st_dir = df["ST_Direction"].iloc[i - 1]
            atr = df["ATR"].iloc[i]

            if pd.isna(ema20) or pd.isna(st) or pd.isna(atr) or atr <= 0:
                continue
            if pd.isna(prev_st_dir):
                continue

            # SuperTrend flip detection
            st_flip_bullish = (st_dir == 1) and (prev_st_dir == -1)
            st_flip_bearish = (st_dir == -1) and (prev_st_dir == 1)

            # ── BUY: ST flips bullish + EMAs aligned up ──
            if st_flip_bullish and ema9 > ema20 and close > ema9 and close > ema20:
                sl = st - 0.5 * atr  # just below SuperTrend level
                target = close + 2.0 * atr

                risk = close - sl
                if risk <= 0:
                    continue
                if target < close + 1.5 * risk:
                    target = close + 1.5 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: ST flipped bullish, EMA9({ema9:.0f})>EMA20({ema20:.0f}), "
                    f"Close({close:.0f})>ST({st:.0f})"
                )
                continue

            # ── SELL: ST flips bearish + EMAs aligned down ──
            if st_flip_bearish and ema9 < ema20 and close < ema9 and close < ema20:
                sl = st + 0.5 * atr  # just above SuperTrend level
                target = close - 2.0 * atr

                risk = sl - close
                if risk <= 0:
                    continue
                if target > close - 1.5 * risk:
                    target = close - 1.5 * risk

                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: ST flipped bearish, EMA9({ema9:.0f})<EMA20({ema20:.0f}), "
                    f"Close({close:.0f})<ST({st:.0f})"
                )

        return df
