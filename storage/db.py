import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DB_PATH = Path("data") / "trading.db"


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    """
    Simple connection factory for SQLite.

    We open a short‑lived connection per operation to avoid
    cross‑thread issues. For a production system, you'd likely
    move to a pooled connection manager or a different database.
    """
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all required tables if they do not exist."""
    with get_conn() as conn:
        c = conn.cursor()

        # Raw tick data (Binance trades)
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL
            )
            """
        )

        # Resampled OHLCV bars for different intervals and sources (live/uploaded)
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,        -- e.g. '1s','1min','5min'
                open_time_ms INTEGER NOT NULL, -- period start timestamp (ms)
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'live',
                UNIQUE(symbol, interval, open_time_ms, source)
            )
            """
        )

        # Optional table to persist basic analytics or backtests if needed later.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                symbol_x TEXT NOT NULL,
                symbol_y TEXT NOT NULL,
                interval TEXT NOT NULL,
                params_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                csv_path TEXT
            )
            """
        )


def insert_tick(ts_ms: int, symbol: str, price: float, size: float) -> None:
    """Insert a single trade tick."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ticks (ts_ms, symbol, price, size) VALUES (?, ?, ?, ?)",
            (int(ts_ms), symbol.upper(), float(price), float(size)),
        )


def fetch_ticks(
    symbol: str,
    since_ms: Optional[int] = None,
    limit: int = 10_000,
) -> List[sqlite3.Row]:
    """
    Fetch recent ticks for a symbol, optionally since a given timestamp.
    """
    symbol = symbol.upper()
    with get_conn() as conn:
        if since_ms is not None:
            cur = conn.execute(
                """
                SELECT ts_ms, symbol, price, size
                FROM ticks
                WHERE symbol = ? AND ts_ms >= ?
                ORDER BY ts_ms ASC
                LIMIT ?
                """,
                (symbol, since_ms, limit),
            )
        else:
            cur = conn.execute(
                """
                SELECT ts_ms, symbol, price, size
                FROM ticks
                WHERE symbol = ?
                ORDER BY ts_ms DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
        rows = cur.fetchall()
        # If we fetched in descending order, reverse for time series order.
        if since_ms is None:
            return list(reversed(rows))
        return rows


def upsert_ohlc_rows(
    rows: Iterable[Tuple[str, str, int, float, float, float, float, float, str]]
) -> None:
    """
    Insert or replace OHLCV bars.

    Row schema:
      (symbol, interval, open_time_ms, open, high, low, close, volume, source)
    """
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO ohlc (
                symbol, interval, open_time_ms,
                open, high, low, close, volume, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time_ms, source)
            DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            list(rows),
        )


def fetch_ohlc(
    symbol: str,
    interval: str,
    limit: int = 500,
    source: str = "live",
) -> List[sqlite3.Row]:
    """Fetch recent OHLCV bars for a symbol/interval/source."""
    symbol = symbol.upper()
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT symbol, interval, open_time_ms, open, high, low, close, volume, source
            FROM ohlc
            WHERE symbol = ? AND interval = ? AND source = ?
            ORDER BY open_time_ms DESC
            LIMIT ?
            """,
            (symbol, interval, source, limit),
        )
        rows = cur.fetchall()
        return list(reversed(rows))


def get_last_ohlc_open_time(
    symbol: str,
    interval: str,
    source: str = "live",
) -> Optional[int]:
    """Return the latest open_time_ms for a given symbol/interval/source, if any."""
    symbol = symbol.upper()
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT MAX(open_time_ms) AS max_open_time
            FROM ohlc
            WHERE symbol = ? AND interval = ? AND source = ?
            """,
            (symbol, interval, source),
        )
        row = cur.fetchone()
        if row and row["max_open_time"] is not None:
            return int(row["max_open_time"])
        return None



