"""
Strategy 2: EMA + VWAP + SuperTrend
────────────────────────────────────────────────────────
LOGIC:
  BUY when:
    - SuperTrend is bullish (direction = 1)
    - Price is above EMA 21 (short-term trend)
    - Price is above VWAP (institutional buying)
    - Either: SuperTrend just flipped bullish, OR
      price pulled back to near VWAP/EMA21 and bounced

  SELL when:
    - SuperTrend is bearish (direction = -1)
    - Price is below EMA 21
    - Price is below VWAP
    - Either: SuperTrend just flipped bearish, OR
      price pulled back to near VWAP/EMA21 and rejected

  This is a trend-following strategy optimized for strong moves.
  SL: SuperTrend level | Target: ATR-based
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyEmaVwapSupertrend(BaseStrategy):

    @property
    def name(self) -> str:
        return "EMA + VWAP + SuperTrend"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Signal"] = "HOLD"
        df["SL"] = np.nan
        df["Target"] = np.nan
        df["Reason"] = ""

        time_ok = self._time_filter(df)

        # Track if we already signaled in the current ST direction
        # to avoid repeat signals on every candle
        last_signal_st_dir = 0

        for i in range(2, len(df)):
            if not time_ok.iloc[i]:
                continue

            close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]
            prev2_close = df["Close"].iloc[i - 2]
            ema9 = df["EMA_9"].iloc[i]
            ema21 = df["EMA_21"].iloc[i]
            vwap = df["VWAP"].iloc[i]
            st = df["SuperTrend"].iloc[i]
            st_dir = df["ST_Direction"].iloc[i]
            prev_st_dir = df["ST_Direction"].iloc[i - 1]
            atr = df["ATR"].iloc[i]

            if pd.isna(st) or pd.isna(atr) or pd.isna(ema21) or pd.isna(vwap):
                continue

            # Reset signal tracker on direction change
            if st_dir != prev_st_dir:
                last_signal_st_dir = 0

            st_flip_bull = (st_dir == 1) and (prev_st_dir == -1)
            st_flip_bear = (st_dir == -1) and (prev_st_dir == 1)

            # Pullback detection: price touched near EMA21/VWAP then moved away
            near_ema21 = abs(prev_close - ema21) < atr * 0.5
            near_vwap = abs(prev_close - vwap) < atr * 0.5

            # ── BUY CONDITIONS ──
            st_bullish = st_dir == 1
            above_ema21 = close > ema21
            above_vwap = close > vwap

            # Trigger: ST flip OR pullback bounce in bullish trend
            flip_trigger = st_flip_bull and above_ema21
            pullback_trigger = (
                st_bullish
                and above_ema21
                and above_vwap
                and (near_ema21 or near_vwap)
                and close > prev_close  # bouncing up
                and last_signal_st_dir != 1
            )

            if flip_trigger or pullback_trigger:
                sl = max(st, close - config.SL_ATR_MULTIPLIER * atr)
                target = close + config.TARGET_ATR_MULTIPLIER * atr
                trigger_type = "ST flip bullish" if st_flip_bull else "Pullback bounce"

                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: {trigger_type}, ST bullish, "
                    f"Close({close:.0f})>EMA21({ema21:.0f}), VWAP({vwap:.0f})"
                )
                last_signal_st_dir = 1
                continue

            # ── SELL CONDITIONS ──
            st_bearish = st_dir == -1
            below_ema21 = close < ema21
            below_vwap = close < vwap

            flip_trigger_sell = st_flip_bear and below_ema21
            pullback_trigger_sell = (
                st_bearish
                and below_ema21
                and below_vwap
                and (near_ema21 or near_vwap)
                and close < prev_close  # rejecting down
                and last_signal_st_dir != -1
            )

            if flip_trigger_sell or pullback_trigger_sell:
                sl = min(st, close + config.SL_ATR_MULTIPLIER * atr)
                target = close - config.TARGET_ATR_MULTIPLIER * atr
                trigger_type = "ST flip bearish" if st_flip_bear else "Pullback rejection"

                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: {trigger_type}, ST bearish, "
                    f"Close({close:.0f})<EMA21({ema21:.0f}), VWAP({vwap:.0f})"
                )
                last_signal_st_dir = -1

        return df
