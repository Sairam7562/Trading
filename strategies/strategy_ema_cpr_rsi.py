"""
Strategy 1: EMA 50 + CPR + RSI
────────────────────────────────────────────────────────
LOGIC:
  BUY when:
    - Price is above EMA 50 (trend filter)
    - Price is above the Top CPR (TC) — bullish zone
    - RSI crosses above oversold (35) from below — momentum confirmation
    - Price is above VWAP (institutional bias)

  SELL when:
    - Price is below EMA 50
    - Price is below Bottom CPR (BC) — bearish zone
    - RSI crosses below overbought (65) from above
    - Price is below VWAP

  SL/Target: ATR-based
"""

import numpy as np
import pandas as pd

import config
from strategies.base import BaseStrategy


class StrategyEmaCprRsi(BaseStrategy):

    @property
    def name(self) -> str:
        return "EMA50 + CPR + RSI"

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
            prev_close = df["Close"].iloc[i - 1]
            ema50 = df["EMA_50"].iloc[i]
            tc = df["TC"].iloc[i]
            bc = df["BC"].iloc[i]
            rsi = df["RSI"].iloc[i]
            prev_rsi = df["RSI"].iloc[i - 1]
            vwap = df["VWAP"].iloc[i]
            atr = df["ATR"].iloc[i]

            if pd.isna(rsi) or pd.isna(atr) or pd.isna(ema50) or pd.isna(vwap):
                continue

            # ── BUY CONDITIONS ──
            trend_bullish = close > ema50
            above_cpr = close > tc
            rsi_buy = rsi > config.RSI_OVERSOLD and prev_rsi <= config.RSI_OVERSOLD
            rsi_zone = config.RSI_OVERSOLD < rsi < config.RSI_OVERBOUGHT
            above_vwap = close > vwap

            # Relaxed RSI: either crossing above oversold OR in neutral zone with strong trend
            rsi_ok = rsi_buy or (rsi_zone and trend_bullish and above_cpr)

            if trend_bullish and above_cpr and rsi_ok and above_vwap:
                sl = close - config.SL_ATR_MULTIPLIER * atr
                target = close + config.TARGET_ATR_MULTIPLIER * atr
                df.iloc[i, df.columns.get_loc("Signal")] = "BUY"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"BUY: Close({close:.0f})>EMA50({ema50:.0f}), "
                    f"above TC({tc:.0f}), RSI={rsi:.0f}, above VWAP({vwap:.0f})"
                )
                continue

            # ── SELL CONDITIONS ──
            trend_bearish = close < ema50
            below_cpr = close < bc
            rsi_sell = rsi < config.RSI_OVERBOUGHT and prev_rsi >= config.RSI_OVERBOUGHT
            rsi_ok_sell = rsi_sell or (rsi_zone and trend_bearish and below_cpr)
            below_vwap = close < vwap

            if trend_bearish and below_cpr and rsi_ok_sell and below_vwap:
                sl = close + config.SL_ATR_MULTIPLIER * atr
                target = close - config.TARGET_ATR_MULTIPLIER * atr
                df.iloc[i, df.columns.get_loc("Signal")] = "SELL"
                df.iloc[i, df.columns.get_loc("SL")] = round(sl, 2)
                df.iloc[i, df.columns.get_loc("Target")] = round(target, 2)
                df.iloc[i, df.columns.get_loc("Reason")] = (
                    f"SELL: Close({close:.0f})<EMA50({ema50:.0f}), "
                    f"below BC({bc:.0f}), RSI={rsi:.0f}, below VWAP({vwap:.0f})"
                )

        return df
