"""
Strategy 3: CPR + VWAP + EMA Crossover (with Narrow CPR filter)
────────────────────────────────────────────────────────
LOGIC:
  This strategy combines CPR breakout with EMA crossover confirmation.
  It works best on trending days (detected via narrow CPR).

  BUY when:
    - CPR is narrow (< 0.3% of pivot — expecting trending day)
    - EMA 9 crosses above EMA 21 (golden crossover on 5-min)
    - Price breaks above Top CPR (TC)
    - Price is above VWAP
    - Close > EMA 50 (higher timeframe trend filter)

  SELL when:
    - CPR is narrow
    - EMA 9 crosses below EMA 21 (death cross on 5-min)
    - Price breaks below Bottom CPR (BC)
    - Price is below VWAP
    - Close < EMA 50

  VARIANT A (aggressive): CPR width < 0.5%
  VARIANT B (conservative): CPR width < 0.3%

  SL: ATR-based | Target: ATR-based with 1:2 RR
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyCprVwapEmaCrossover(BaseStrategy):

    def __init__(self, cpr_width_threshold: float = 0.3):
        self.cpr_width_threshold = cpr_width_threshold

    @property
    def name(self) -> str:
        return f"CPR + VWAP + EMA Crossover (CPR<{self.cpr_width_threshold}%)"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Signal"] = "HOLD"
        df["SL"] = np.nan
        df["Target"] = np.nan
        df["Reason"] = ""

        time_ok = self._time_filter(df)

        for i in range(1, len(df)):
            if not time_ok.iloc[i]:
                continue

            close = df["Close"].iloc[i]
            ema9 = df["EMA_9"].iloc[i]
            ema21 = df["EMA_21"].iloc[i]
            prev_ema9 = df["EMA_9"].iloc[i - 1]
            prev_ema21 = df["EMA_21"].iloc[i - 1]
            ema50 = df["EMA_50"].iloc[i]
            tc = df["TC"].iloc[i]
            bc = df["BC"].iloc[i]
            vwap = df["VWAP"].iloc[i]
            atr = df["ATR"].iloc[i]
            cpr_width = df["CPR_Width"].iloc[i]

            if pd.isna(ema50) or pd.isna(atr) or pd.isna(vwap) or pd.isna(cpr_width):
                continue

            # Narrow CPR filter — expecting a trending day
            narrow_cpr = cpr_width < self.cpr_width_threshold

            # EMA crossovers
            ema_cross_up = (ema9 > ema21) and (prev_ema9 <= prev_ema21)
            ema_cross_down = (ema9 < ema21) and (prev_ema9 >= prev_ema21)

            # Also allow already-crossed state with narrow CPR breakout
            ema_bullish = ema9 > ema21
            ema_bearish = ema9 < ema21

            # ── BUY ──
            above_tc = close > tc
            above_vwap = close > vwap
            trend_up = close > ema50

            if narrow_cpr and (ema_cross_up or (ema_bullish and above_tc)) and above_vwap and trend_up:
                sl = close - config.SL_ATR_MULTIPLIER * atr
                target = close + config.TARGET_ATR_MULTIPLIER * atr

                reason_trigger = "EMA9x21 cross UP" if ema_cross_up else "EMA9>21 + TC breakout"
                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: Narrow CPR({cpr_width:.2f}%), {reason_trigger}, "
                    f"above VWAP({vwap:.0f}), Close({close:.0f})>EMA50({ema50:.0f})"
                )
                continue

            # ── SELL ──
            below_bc = close < bc
            below_vwap = close < vwap
            trend_down = close < ema50

            if narrow_cpr and (ema_cross_down or (ema_bearish and below_bc)) and below_vwap and trend_down:
                sl = close + config.SL_ATR_MULTIPLIER * atr
                target = close - config.TARGET_ATR_MULTIPLIER * atr

                reason_trigger = "EMA9x21 cross DOWN" if ema_cross_down else "EMA9<21 + BC breakdown"
                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: Narrow CPR({cpr_width:.2f}%), {reason_trigger}, "
                    f"below VWAP({vwap:.0f}), Close({close:.0f})<EMA50({ema50:.0f})"
                )

        return df
