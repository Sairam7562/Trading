"""
Technical Indicator Engine
Computes all indicators needed for intraday strategies on 5-min charts.
Indicators: EMA, CPR, VWAP, RSI, SuperTrend, ATR
"""

import numpy as np
import pandas as pd

import config


# ──────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA."""
    return series.ewm(span=period, adjust=False).mean()


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 9, 20, 21, 50 to the dataframe."""
    df = df.copy()
    df["EMA_9"] = ema(df["Close"], config.EMA_FAST)
    df["EMA_20"] = ema(df["Close"], config.EMA_20)
    df["EMA_21"] = ema(df["Close"], config.EMA_MID)
    df["EMA_50"] = ema(df["Close"], config.EMA_SLOW)
    return df


# ──────────────────────────────────────────────────────────
# CPR (Central Pivot Range)
# ──────────────────────────────────────────────────────────

def add_cpr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CPR from previous day's High, Low, Close.
    CPR has 3 levels:
      - Pivot (P) = (PrevHigh + PrevLow + PrevClose) / 3
      - BC (Bottom CPR) = (PrevHigh + PrevLow) / 2
      - TC (Top CPR) = 2*P - BC
    Also adds S1, S2, R1, R2 pivot levels.
    """
    df = df.copy()

    H = df["PrevHigh"]
    L = df["PrevLow"]
    C = df["PrevClose"]

    df["Pivot"] = (H + L + C) / 3
    df["BC"] = (H + L) / 2
    df["TC"] = 2 * df["Pivot"] - df["BC"]

    # Ensure TC > BC (swap if needed)
    tc = df["TC"].copy()
    bc = df["BC"].copy()
    df["TC"] = np.maximum(tc, bc)
    df["BC"] = np.minimum(tc, bc)

    # Support & Resistance levels
    df["R1"] = 2 * df["Pivot"] - L
    df["S1"] = 2 * df["Pivot"] - H
    df["R2"] = df["Pivot"] + (H - L)
    df["S2"] = df["Pivot"] - (H - L)

    # CPR Width (narrow CPR = trending day expected)
    df["CPR_Width"] = (df["TC"] - df["BC"]) / df["Pivot"] * 100

    return df


# ──────────────────────────────────────────────────────────
# VWAP (Volume Weighted Average Price)
# ──────────────────────────────────────────────────────────

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday VWAP — resets each day.
    VWAP = cumsum(TP * Volume) / cumsum(Volume)
    """
    df = df.copy()
    df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Group by trading day
    df["_date"] = df.index.date
    grouped = df.groupby("_date")

    vwap_values = []
    for _, group in grouped:
        cum_vol = group["Volume"].cumsum()
        cum_tp_vol = (group["TP"] * group["Volume"]).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap_values.append(vwap)

    df["VWAP"] = pd.concat(vwap_values)
    df.drop(columns=["TP", "_date"], inplace=True)

    return df


# ──────────────────────────────────────────────────────────
# RSI (Relative Strength Index)
# ──────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """Wilder's RSI."""
    if period is None:
        period = config.RSI_PERIOD

    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# ──────────────────────────────────────────────────────────
# ATR (Average True Range)
# ──────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """Wilder's ATR."""
    if period is None:
        period = config.ATR_PERIOD

    df = df.copy()
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return df


# ──────────────────────────────────────────────────────────
# SuperTrend
# ──────────────────────────────────────────────────────────

def add_supertrend(df: pd.DataFrame, period: int = None, multiplier: float = None) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Returns columns: SuperTrend, ST_Direction (1=bullish, -1=bearish)
    """
    if period is None:
        period = config.SUPERTREND_PERIOD
    if multiplier is None:
        multiplier = config.SUPERTREND_MULTIPLIER

    df = df.copy()

    # Ensure ATR exists
    if "ATR" not in df.columns:
        df = add_atr(df, period)

    hl2 = (df["High"] + df["Low"]) / 2
    atr = df["ATR"]

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        # Lower band logic
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["Close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            pass  # keep current lower_band
        else:
            lower_band.iloc[i] = lower_band.iloc[i - 1]

        # Upper band logic
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["Close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            pass  # keep current upper_band
        else:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        # Direction
        if supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            if df["Close"].iloc[i] > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if df["Close"].iloc[i] < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1

    df["SuperTrend"] = supertrend
    df["ST_Direction"] = direction

    return df


# ──────────────────────────────────────────────────────────
# Master function — add all indicators at once
# ──────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator to the dataframe."""
    df = add_emas(df)
    df = add_cpr(df)
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_supertrend(df)
    return df
