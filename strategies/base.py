"""
Base Strategy class — all strategies inherit from this.
Defines the interface for signal generation.
"""

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd

import config


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    signal: Signal
    entry_price: float
    stop_loss: float
    target: float
    reason: str


class BaseStrategy(ABC):
    """
    Every strategy must implement:
      - name: str property
      - generate_signals(df) -> df with 'Signal', 'SL', 'Target' columns
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns to df:
          - Signal: 'BUY', 'SELL', or 'HOLD'
          - SL: stop-loss price
          - Target: target price
          - Reason: text description of why signal fired
        """
        ...

    def _time_filter(self, df: pd.DataFrame) -> pd.Series:
        """Returns boolean mask: True if within allowed trading hours."""
        times = df.index.time
        no_trade_after = dt.time(
            int(config.NO_NEW_TRADE_AFTER.split(":")[0]),
            int(config.NO_NEW_TRADE_AFTER.split(":")[1]),
        )
        market_open = dt.time(9, 20)  # skip first candle (volatile)
        return pd.Series(
            [(t >= market_open) and (t <= no_trade_after) for t in times],
            index=df.index,
        )
