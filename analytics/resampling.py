from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional

import pandas as pd

from storage.db import fetch_ticks, get_last_ohlc_open_time, upsert_ohlc_rows


INTERVAL_MAPPING: Dict[str, str] = {
    "1s": "1S",
    "1sec": "1S",
    "1": "1S",
    "1m": "1T",
    "1min": "1T",
    "5m": "5T",
}


def _ticks_to_dataframe(ticks) -> pd.DataFrame:
    """Convert DB tick rows into a time‑indexed DataFrame."""
    if not ticks:
        return pd.DataFrame(columns=["price", "size"])
    df = pd.DataFrame(ticks)
    df["ts_ms"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("ts_ms").sort_index()
    return df[["price", "size"]]


def resample_ticks_for_symbol(
    symbol: str,
    interval: str,
    source: str = "live",
    lookback_ms: int = 60 * 60 * 1000,
) -> None:
    """
    Resample recent ticks for a symbol into OHLCV bars and persist them.

    This function:
      - Looks up the last stored bar time for (symbol, interval, source)
      - Fetches ticks since that time (or a lookback window)
      - Uses pandas.resample to create OHLCV
      - Upserts the resulting bars into the `ohlc` table
    """
    interval = interval.lower()
    if interval not in INTERVAL_MAPPING:
        raise ValueError(f"Unsupported interval: {interval}")

    pandas_rule = INTERVAL_MAPPING[interval]

    last_open_time = get_last_ohlc_open_time(symbol, interval, source=source)
    since_ms: Optional[int]
    if last_open_time is not None:
        # Start from the last bar start to avoid gaps/overlaps.
        since_ms = last_open_time
    else:
        # No existing bars; pull a reasonable lookback window of raw ticks.
        now_ms = int(dt.datetime.utcnow().timestamp() * 1000)
        since_ms = max(0, now_ms - lookback_ms)

    ticks = fetch_ticks(symbol, since_ms=since_ms, limit=100_000)
    df = _ticks_to_dataframe(ticks)
    if df.empty:
        return

    ohlc = df["price"].resample(pandas_rule).ohlc()
    vol = df["size"].resample(pandas_rule).sum().rename("volume")
    merged = ohlc.join(vol, how="inner").dropna()
    if merged.empty:
        return

    rows = []
    for ts, row in merged.iterrows():
        open_time_ms = int(ts.timestamp() * 1000)
        rows.append(
            (
                symbol.upper(),
                interval,
                open_time_ms,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                source,
            )
        )

    if rows:
        upsert_ohlc_rows(rows)



