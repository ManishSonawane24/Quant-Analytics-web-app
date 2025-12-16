from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.Series:
    """Simple log returns of a price series."""
    return np.log(close).diff()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z‑score: (x - rolling_mean) / rolling_std.

    This is a standard way to normalize a spread and measure how many
    standard deviations it is from its recent mean, used heavily in
    mean‑reversion/stat‑arb strategies.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=1)
    z = (series - rolling_mean) / rolling_std
    return z


def rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> pd.Series:
    """
    Rolling Pearson correlation between two series.
    
    Computes rolling correlation using a simple, reliable approach.
    """
    # Align the series first
    x_aligned, y_aligned = x.align(y, join="inner")
    if x_aligned.empty or y_aligned.empty:
        return pd.Series(dtype=float, index=x_aligned.index if not x_aligned.empty else y_aligned.index)
    
    # Use the simplest approach: rolling window on a DataFrame
    df = pd.DataFrame({"x": x_aligned, "y": y_aligned})
    # Compute rolling correlation - this returns a Series with correlation values
    rolling_corr = df["x"].rolling(window=window).corr(df["y"])
    return rolling_corr


@dataclass
class PriceStats:
    mean_price: float
    std_price: float
    last_price: float
    last_return: float
    volatility_annualized: float


def compute_price_stats(close: pd.Series, trading_hours_per_year: int = 24 * 365) -> Optional[PriceStats]:
    """
    Compute basic descriptive statistics for a close series.

    The volatility is annualized from the standard deviation of returns,
    assuming evenly spaced bars representing `trading_hours_per_year`.
    """
    if close.empty:
        return None
    ret = compute_returns(close).dropna()
    if ret.empty:
        last_ret = 0.0
        vol_ann = 0.0
    else:
        last_ret = float(ret.iloc[-1])
        vol_ann = float(ret.std(ddof=1) * np.sqrt(trading_hours_per_year))

    return PriceStats(
        mean_price=float(close.mean()),
        std_price=float(close.std(ddof=1)),
        last_price=float(close.iloc[-1]),
        last_return=last_ret,
        volatility_annualized=vol_ann,
    )


def mean_reversion_backtest(
    spread: pd.Series,
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    notional: float = 1.0,
) -> pd.DataFrame:
    """
    Simple mean‑reversion backtest on a spread using z‑score signals.

    Trading logic (one‑unit spread position for intuition):
      - When z > entry_z: go SHORT spread (expect spread to fall)
      - When z < -entry_z: go LONG spread (expect spread to rise)
      - When |z| < exit_z: flat (exit any open position)

    PnL is computed on spread changes multiplied by the position.
    """
    df = pd.DataFrame({"spread": spread, "z": zscore}).dropna()
    if df.empty:
        return pd.DataFrame()

    position = np.zeros(len(df))

    for i in range(1, len(df)):
        prev_pos = position[i - 1]
        z_val = df["z"].iloc[i]
        if z_val > entry_z:
            position[i] = -1.0
        elif z_val < -entry_z:
            position[i] = 1.0
        elif abs(z_val) < exit_z:
            position[i] = 0.0
        else:
            position[i] = prev_pos

    df["position"] = position * notional
    df["spread_change"] = df["spread"].diff().fillna(0.0)
    df["pnl"] = -(df["position"].shift(1).fillna(0.0)) * df["spread_change"]
    df["equity"] = df["pnl"].cumsum()
    return df



