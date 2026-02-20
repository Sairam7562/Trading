"""
Data fetcher for NIFTY and BANKNIFTY 5-minute intraday data.
Supports: yfinance (free), with hooks for Zerodha Kite API.
"""

import datetime as dt
import pandas as pd
import numpy as np
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

import config


def fetch_yfinance(symbol_key: str) -> pd.DataFrame:
    """
    Download 5-min candle data from Yahoo Finance.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    indexed by datetime (IST).
    """
    ticker = config.SYMBOLS[symbol_key]
    end = dt.datetime.now()
    start = end - dt.timedelta(days=config.LOOKBACK_DAYS)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=config.INTERVAL,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol_key} ({ticker})")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure standard column names
    df = df.rename(columns={
        "Adj Close": "Close",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Convert index to IST if needed
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df.index.name = "Datetime"

    # Filter market hours only
    df = _filter_market_hours(df)

    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows between market open and close."""
    times = df.index.time
    market_open = dt.time(9, 15)
    market_close = dt.time(15, 30)
    mask = (times >= market_open) & (times <= market_close)
    return df.loc[mask].copy()


def add_daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous-day OHLC columns needed for CPR calculation.
    Groups by date, takes the previous trading day's OHLC.
    """
    df = df.copy()
    df["Date"] = df.index.date

    daily = df.groupby("Date").agg(
        DayOpen=("Open", "first"),
        DayHigh=("High", "max"),
        DayLow=("Low", "min"),
        DayClose=("Close", "last"),
    )

    # Shift to get *previous* day values
    daily["PrevHigh"] = daily["DayHigh"].shift(1)
    daily["PrevLow"] = daily["DayLow"].shift(1)
    daily["PrevClose"] = daily["DayClose"].shift(1)

    df = df.merge(daily[["PrevHigh", "PrevLow", "PrevClose"]], left_on="Date", right_index=True, how="left")
    df.drop(columns=["Date"], inplace=True)

    # Drop the first day (no previous day data)
    df.dropna(subset=["PrevHigh", "PrevLow", "PrevClose"], inplace=True)

    return df


def generate_synthetic_data(symbol_key: str, days: int = 50) -> pd.DataFrame:
    """
    Generate realistic synthetic 5-min OHLCV data for backtesting
    when live data is unavailable. Mimics NIFTY/BANKNIFTY behavior.
    """
    np.random.seed(42 if symbol_key == "NIFTY" else 99)

    base_price = 22000.0 if symbol_key == "NIFTY" else 48000.0
    volatility = 0.0008 if symbol_key == "NIFTY" else 0.0012

    records = []
    price = base_price

    start_date = dt.date.today() - dt.timedelta(days=days + 20)
    current_date = start_date
    trading_days = 0

    while trading_days < days:
        current_date += dt.timedelta(days=1)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue
        trading_days += 1

        # Generate 5-min candles from 09:15 to 15:25
        t = dt.time(9, 15)
        day_open = price
        # Add gap open (common in Indian markets)
        gap = np.random.normal(0, base_price * 0.003)
        price = day_open + gap

        while t < dt.time(15, 30):
            timestamp = dt.datetime.combine(current_date, t)

            # Mean-reverting random walk with intraday patterns
            hour = t.hour + t.minute / 60.0
            # Higher volatility at open and close
            time_vol = volatility * (1.5 if hour < 10 or hour > 14.5 else 1.0)

            returns = np.random.normal(0, time_vol)
            # Add slight trend bias based on day
            trend = np.random.normal(0, 0.0001)
            price *= (1 + returns + trend)

            # Generate OHLC from the base price
            high = price * (1 + abs(np.random.normal(0, volatility * 0.5)))
            low = price * (1 - abs(np.random.normal(0, volatility * 0.5)))
            open_price = price * (1 + np.random.normal(0, volatility * 0.3))
            close_price = price * (1 + np.random.normal(0, volatility * 0.3))

            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            volume = int(np.random.lognormal(mean=12, sigma=0.8))

            records.append({
                "Datetime": timestamp,
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
            })

            # Increment by 5 minutes
            mins = t.hour * 60 + t.minute + 5
            if mins >= 15 * 60 + 30:
                break
            t = dt.time(mins // 60, mins % 60)

    df = pd.DataFrame(records)
    df.set_index("Datetime", inplace=True)

    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a TradingView-exported CSV file.
    Expected columns: time, open, high, low, close, Volume
    (case-insensitive matching).
    """
    import os
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names to title case
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("time", "datetime", "date"):
            col_map[c] = "Datetime"
        elif cl == "open":
            col_map[c] = "Open"
        elif cl == "high":
            col_map[c] = "High"
        elif cl == "low":
            col_map[c] = "Low"
        elif cl == "close":
            col_map[c] = "Close"
        elif cl == "volume":
            col_map[c] = "Volume"
    df.rename(columns=col_map, inplace=True)

    required = ["Datetime", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    # Ensure Volume column exists
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Filter market hours
    df = _filter_market_hours(df)

    print(f"  [DATA] Loaded {len(df)} candles from CSV: {csv_path}")
    return df


def load_data(symbol_key: str, use_synthetic: bool = False, csv_path: str = None) -> pd.DataFrame:
    """
    Load 5-min data for the given symbol.
    Priority: csv_path > yfinance > synthetic.
    """
    if csv_path:
        print(f"  [DATA] Loading from CSV: {csv_path}...")
        df = load_csv(csv_path)
    elif use_synthetic:
        print(f"  [DATA] Generating synthetic data for {symbol_key}...")
        df = generate_synthetic_data(symbol_key)
    else:
        if not HAS_YFINANCE:
            print(f"  [DATA] yfinance not installed, using synthetic data...")
            df = generate_synthetic_data(symbol_key)
        else:
            try:
                print(f"  [DATA] Fetching {symbol_key} from Yahoo Finance...")
                df = fetch_yfinance(symbol_key)
                if len(df) < 100:
                    raise ValueError("Insufficient data points")
                print(f"  [DATA] Got {len(df)} candles for {symbol_key}")
            except Exception as e:
                print(f"  [DATA] yfinance failed ({e}), using synthetic data...")
                df = generate_synthetic_data(symbol_key)

    df = add_daily_ohlc(df)
    print(f"  [DATA] {symbol_key}: {len(df)} candles across {df.index.date[-1] - df.index.date[0]} days")
    return df
