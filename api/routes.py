from __future__ import annotations

import asyncio
import io
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from alerts.rules import AlertRule, alert_registry
from analytics import adf as adf_mod
from analytics import regression as reg_mod
from analytics import resampling as resample_mod
from analytics import stats as stats_mod
from ingestion.websocket_client import BinanceFuturesClient, Trade
from storage import db as db_mod


DEFAULT_SYMBOLS = ["btcusdt", "ethusdt"]
RESAMPLE_INTERVALS = ["1s", "1m", "5m"]

app = FastAPI(title="Quant Analytics Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Simple root endpoint to avoid 404s when hitting '/'.
    """
    return {
        "service": "quant-analytics-backend",
        "status": "ok",
        "message": "Use /docs for API, Streamlit UI on http://localhost:8501",
    }


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    """
    Return an empty favicon to avoid noisy 404s in logs.
    """
    return JSONResponse(content=None, status_code=204)


@app.on_event("startup")
async def on_startup() -> None:
    """
    Initialize DB and start background tasks for:
      - Binance WebSocket ingestion
      - Periodic resampling of ticks into OHLCV bars
    """
    db_mod.init_db()

    loop = asyncio.get_event_loop()

    async def on_trade(trade: Trade) -> None:
        db_mod.insert_tick(trade.timestamp, trade.symbol, trade.price, trade.size)

    client = BinanceFuturesClient(symbols=DEFAULT_SYMBOLS, on_trade=on_trade)

    async def ingest_loop():
        await client.run_forever()

    async def resample_loop():
        while True:
            try:
                for sym in DEFAULT_SYMBOLS:
                    for interval in RESAMPLE_INTERVALS:
                        try:
                            resample_mod.resample_ticks_for_symbol(sym, interval)
                        except Exception as e:
                            # Log but don't stop the loop
                            print(f"Resampling error for {sym} {interval}: {e}")
            except Exception as e:
                print(f"Resample loop error: {e}")
            await asyncio.sleep(1.0)

    loop.create_task(ingest_loop())
    loop.create_task(resample_loop())


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "time": time.time()}


@app.get("/symbols")
async def list_symbols() -> Dict[str, List[str]]:
    return {"symbols": [s.upper() for s in DEFAULT_SYMBOLS]}


@app.get("/data/status")
async def data_status() -> Dict[str, Any]:
    """
    Diagnostic endpoint to check data ingestion status.
    Returns counts of ticks and OHLC bars for each symbol and interval.
    """
    status: Dict[str, Any] = {
        "ticks": {},
        "ohlc": {},
    }
    
    for sym in DEFAULT_SYMBOLS:
        sym_upper = sym.upper()
        # Count ticks
        ticks = db_mod.fetch_ticks(sym_upper, limit=1)
        status["ticks"][sym_upper] = len(ticks) > 0
        
        # Count OHLC bars for each interval
        status["ohlc"][sym_upper] = {}
        for interval in RESAMPLE_INTERVALS:
            ohlc_rows = db_mod.fetch_ohlc(sym_upper, interval, limit=1)
            status["ohlc"][sym_upper][interval] = len(ohlc_rows)
    
    return status


def _rows_to_ohlc_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


@app.get("/ohlc")
async def get_ohlc(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    source: str = "live",
) -> Dict[str, Any]:
    rows = db_mod.fetch_ohlc(symbol, interval, limit=limit, source=source)
    df = _rows_to_ohlc_df(rows)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "source": source,
        "data": df.reset_index().to_dict(orient="records"),
    }


@app.get("/analytics/pair")
async def analytics_pair(
    symbol_y: str,
    symbol_x: str,
    interval: str = "1m",
    window: int = 50,
    regression: reg_mod.RegressionType = "ols",
    limit: int = 500,
    run_adf: bool = False,
) -> Dict[str, Any]:
    """
    Compute pair analytics for (symbol_y, symbol_x):
      - Hedge ratio (OLS/robust/Kalman)
      - Spread
      - Z‑score (rolling)
      - Rolling correlation
      - Optional ADF test on spread
      - Trigger any matching alert rules
    """
    try:
        rows_y = db_mod.fetch_ohlc(symbol_y, interval, limit=limit)
        rows_x = db_mod.fetch_ohlc(symbol_x, interval, limit=limit)
        df_y = _rows_to_ohlc_df(rows_y)
        df_x = _rows_to_ohlc_df(rows_x)

        if df_y.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No OHLC data found for {symbol_y.upper()} at interval {interval}. Please wait for data ingestion or upload data first.",
            )
        if df_x.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No OHLC data found for {symbol_x.upper()} at interval {interval}. Please wait for data ingestion or upload data first.",
            )

        close_y = df_y["close"]
        close_x = df_x["close"]

        # Align the series and check if we have enough overlapping data
        close_y_aligned, close_x_aligned = close_y.align(close_x, join="inner")
        if len(close_y_aligned) < window:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough overlapping data points ({len(close_y_aligned)}) for window size {window}. Need at least {window} points.",
            )

        hedge = reg_mod.build_spread_and_zscore(
            y=close_y, x=close_x, regression_type=regression, window=window
        )

        # Calculate correlation safely
        corr = stats_mod.rolling_correlation(close_y, close_x, window=window)
        stats_y = stats_mod.compute_price_stats(close_y)
        stats_x = stats_mod.compute_price_stats(close_x)

        adf_result: Optional[adf_mod.ADFResult] = None
        if run_adf:
            try:
                adf_result = adf_mod.run_adf(hedge.spread)
            except Exception as e:
                # Don't fail the whole request if ADF fails
                pass

        triggers = alert_registry.evaluate_zscore_alerts(
            symbol_y=symbol_y, symbol_x=symbol_x, zscore=hedge.zscore
        )

        # Prepare time-series payloads with explicit time columns
        spread_df = hedge.spread.reset_index()
        spread_df.columns = ["time", "spread"]
        z_df = hedge.zscore.reset_index()
        z_df.columns = ["time", "z"]
        corr_df = corr.reset_index()
        corr_df.columns = ["time", "corr"]

        payload: Dict[str, Any] = {
            "symbol_y": symbol_y.upper(),
            "symbol_x": symbol_x.upper(),
            "interval": interval,
            "regression": regression,
            "window": window,
            "hedge_ratio": hedge.hedge_ratio,
            "intercept": hedge.intercept,
            "spread": spread_df.to_dict(orient="records"),
            "zscore": z_df.to_dict(orient="records"),
            "correlation": corr_df.to_dict(orient="records"),
            "stats": {
                "y": stats_y.__dict__ if stats_y else None,
                "x": stats_x.__dict__ if stats_x else None,
            },
            "alerts_triggered": [t.__dict__ for t in triggers],
        }

        if adf_result:
            payload["adf"] = adf_result.__dict__

        return payload
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Analytics computation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/analytics/backtest")
async def run_backtest(
    symbol_y: str,
    symbol_x: str,
    interval: str = "1m",
    window: int = 50,
    regression: reg_mod.RegressionType = "ols",
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    limit: int = 1000,
) -> Dict[str, Any]:
    rows_y = db_mod.fetch_ohlc(symbol_y, interval, limit=limit)
    rows_x = db_mod.fetch_ohlc(symbol_x, interval, limit=limit)
    df_y = _rows_to_ohlc_df(rows_y)
    df_x = _rows_to_ohlc_df(rows_x)

    if df_y.empty or df_x.empty:
        raise HTTPException(status_code=400, detail="Not enough data for backtest")

    close_y = df_y["close"]
    close_x = df_x["close"]
    hedge = reg_mod.build_spread_and_zscore(
        y=close_y, x=close_x, regression_type=regression, window=window
    )
    bt = stats_mod.mean_reversion_backtest(
        hedge.spread, hedge.zscore, entry_z=entry_z, exit_z=exit_z
    )
    if bt.empty:
        raise HTTPException(status_code=400, detail="Backtest produced no trades")

    # Persist minimal metadata for later CSV downloads if desired.
    ts_now = int(time.time() * 1000)
    name = f"MR_{symbol_y.upper()}_{symbol_x.upper()}_{interval}_{ts_now}"

    # Serialize to records for UI
    records = bt.reset_index().rename(columns={"index": "time"}).to_dict(orient="records")
    result = {
        "backtest_name": name,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "equity_curve": [
            {"time": str(idx), "equity": float(val)}
            for idx, val in zip(bt.index, bt["equity"])
        ],
        "pnl": float(bt["pnl"].sum()),
        "records": records,
    }
    return result


@app.post("/alerts")
async def create_alert(rule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a simple z‑score alert rule.
    Expected payload:
      {
        "name": "...",
        "symbol_y": "BTCUSDT",
        "symbol_x": "ETHUSDT",
        "metric": "zscore",
        "threshold": 2.0,
        "direction": "above"
      }
    """
    rule_id = str(uuid.uuid4())
    try:
        alert = AlertRule(
            id=rule_id,
            name=rule.get("name", f"Alert {rule_id[:6]}"),
            symbol_y=rule["symbol_y"],
            symbol_x=rule["symbol_x"],
            metric=rule.get("metric", "zscore"),
            threshold=float(rule.get("threshold", 2.0)),
            direction=rule.get("direction", "above"),
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    alert_registry.add_rule(alert)
    return {"id": rule_id, "rule": alert.__dict__}


@app.get("/alerts")
async def list_alerts() -> Dict[str, Any]:
    return {"rules": [r.__dict__ for r in alert_registry.list_rules()]}


@app.delete("/alerts/{rule_id}")
async def delete_alert(rule_id: str) -> Dict[str, Any]:
    alert_registry.remove_rule(rule_id)
    return {"deleted": rule_id}


@app.post("/upload/ohlc")
async def upload_ohlc_csv(
    file: UploadFile = File(...),
    symbol: Optional[str] = None,
    interval: str = "1m",
) -> Dict[str, Any]:
    """
    Ingest an OHLC CSV file and store it in the same `ohlc` table.

    Expected columns:
      - timestamp or open_time
      - open, high, low, close, volume
    """
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    time_col = "timestamp" if "timestamp" in df.columns else "open_time"
    if time_col not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain timestamp/open_time")

    if symbol is None:
        # Try to infer symbol from filename, otherwise mark as GENERIC.
        symbol = file.filename.split(".")[0].upper() if file.filename else "GENERIC"

    ts = pd.to_datetime(df[time_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    open_time_ms = (ts.astype("int64") // 10**6).astype("int64")

    rows = []
    for i, row in df.iterrows():
        rows.append(
            (
                symbol.upper(),
                interval,
                int(open_time_ms.iloc[i]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0.0)),
                "upload",
            )
        )

    db_mod.upsert_ohlc_rows(rows)
    return {"status": "ok", "rows": len(rows)}


@app.get("/export/ohlc")
async def export_ohlc_csv(
    symbol: str,
    interval: str = "1m",
    source: str = "live",
    limit: int = 5000,
) -> StreamingResponse:
    rows = db_mod.fetch_ohlc(symbol, interval, limit=limit, source=source)
    df = _rows_to_ohlc_df(rows).reset_index()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"ohlc_{symbol.upper()}_{interval}_{source}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/export/backtest")
async def export_backtest_csv(payload: Dict[str, Any]) -> StreamingResponse:
    """
    Export backtest results passed from the frontend as CSV.

    This keeps the backend stateless for backtests while still allowing
    users to download exactly what they see in the UI.
    """
    records = payload.get("records")
    if not records:
        raise HTTPException(status_code=400, detail="No records to export")
    df = pd.DataFrame(records)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = payload.get("name", "backtest.csv")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=True)



