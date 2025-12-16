import os
import time
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


@st.cache_data(ttl=5.0)
def fetch_symbols() -> List[str]:
    """Fetch symbols from backend with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{BACKEND_URL}/symbols", timeout=10)
            resp.raise_for_status()
            return resp.json().get("symbols", [])
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise requests.exceptions.Timeout(
                f"Backend at {BACKEND_URL} is not responding. "
                f"Please ensure the backend is running (check if 'python app.py' is running)."
            )
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Cannot connect to backend at {BACKEND_URL}. "
                f"Please ensure the FastAPI backend is running on port 8000."
            )
    return []


def check_backend_health() -> bool:
    """Check if backend is responding."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_data_status() -> Dict[str, Any]:
    """Fetch data ingestion status from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/data/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"ticks": {}, "ohlc": {}}


def fetch_ohlc(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(f"{BACKEND_URL}/ohlc", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]
    if not data:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    return df


def fetch_pair_analytics(
    symbol_y: str,
    symbol_x: str,
    interval: str,
    window: int,
    regression: str,
    run_adf: bool,
) -> Dict[str, Any]:
    params = {
        "symbol_y": symbol_y,
        "symbol_x": symbol_x,
        "interval": interval,
        "window": window,
        "regression": regression,
        "run_adf": str(run_adf).lower(),
    }
    resp = requests.get(f"{BACKEND_URL}/analytics/pair", params=params, timeout=20)
    if resp.status_code != 200:
        error_detail = resp.json().get("detail", f"HTTP {resp.status_code} Error")
        raise requests.HTTPError(f"{error_detail}")
    return resp.json()


def run_backtest_api(
    symbol_y: str,
    symbol_x: str,
    interval: str,
    window: int,
    regression: str,
    entry_z: float,
    exit_z: float,
) -> Dict[str, Any]:
    payload = {
        "symbol_y": symbol_y,
        "symbol_x": symbol_x,
        "interval": interval,
        "window": window,
        "regression": regression,
        "entry_z": entry_z,
        "exit_z": exit_z,
    }
    resp = requests.post(f"{BACKEND_URL}/analytics/backtest", params=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_alert_api(rule: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{BACKEND_URL}/alerts", json=rule, timeout=5)
    resp.raise_for_status()
    return resp.json()


def list_alerts_api() -> List[Dict[str, Any]]:
    resp = requests.get(f"{BACKEND_URL}/alerts", timeout=5)
    resp.raise_for_status()
    return resp.json().get("rules", [])


def upload_ohlc_csv(file, symbol: str, interval: str) -> Dict[str, Any]:
    files = {"file": (file.name, file.getvalue(), "text/csv")}
    params = {"symbol": symbol, "interval": interval}
    resp = requests.post(f"{BACKEND_URL}/upload/ohlc", files=files, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def layout_sidebar(symbols: List[str]) -> Dict[str, Any]:
    st.sidebar.header("Configuration")
    selected = st.sidebar.multiselect(
        "Symbols (multi‑select, pick 2 for pairs):",
        options=symbols,
        default=symbols[:2] if len(symbols) >= 2 else symbols,
    )

    interval = st.sidebar.selectbox("Timeframe", options=["1s", "1m", "5m"], index=1)
    window = st.sidebar.slider("Rolling window (bars)", min_value=20, max_value=500, value=100, step=10)
    regression = st.sidebar.selectbox(
        "Regression type",
        options=["ols", "robust", "kalman"],
        format_func=lambda x: {"ols": "OLS", "robust": "Robust (Huber)", "kalman": "Kalman"}.get(x, x),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alerts")
    alert_name = st.sidebar.text_input("Alert name", value="Z > 2 Alert")
    alert_threshold = st.sidebar.number_input("Z‑score threshold", value=2.0, step=0.1)
    alert_direction = st.sidebar.selectbox("Direction", options=["above", "below"])
    create_alert_btn = st.sidebar.button("Create alert for current pair")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtest")
    entry_z = st.sidebar.number_input("Entry z", value=2.0, step=0.1)
    exit_z = st.sidebar.number_input("Exit z", value=0.0, step=0.1)
    run_backtest = st.sidebar.button("Run mean‑reversion backtest")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Upload")
    upload_symbol = st.sidebar.text_input("Upload symbol (e.g. BTCUSDT)", value="BTCUSDT")
    upload_interval = st.sidebar.selectbox("Upload interval", options=["1s", "1m", "5m"], index=1)
    upload_file = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    upload_btn = st.sidebar.button("Ingest CSV into backend")

    return {
        "selected_symbols": selected,
        "interval": interval,
        "window": window,
        "regression": regression,
        "alert": {
            "name": alert_name,
            "threshold": alert_threshold,
            "direction": alert_direction,
            "create": create_alert_btn,
        },
        "backtest": {
            "entry_z": entry_z,
            "exit_z": exit_z,
            "run": run_backtest,
        },
        "upload": {
            "symbol": upload_symbol,
            "interval": upload_interval,
            "file": upload_file,
            "run": upload_btn,
        },
    }


def main():
    st.set_page_config(
        page_title="Quant Analytics Dashboard",
        layout="wide",
    )
    st.title("Real‑Time Quant Analytics Dashboard (Binance Futures)")
    st.caption("Live tick ingestion, resampled OHLC, pair trading analytics, alerts, and backtesting.")

    # Quick backend health check
    if not check_backend_health():
        st.warning("⚠️ **Backend not responding**. Checking connection...")
        time.sleep(1)
        if not check_backend_health():
            st.error(f"🔌 **Cannot connect to backend at {BACKEND_URL}**")
            st.info("💡 **Please ensure the backend is running**:")
            st.code("python app.py", language="bash")
            if st.button("🔄 Retry Connection"):
                st.rerun()
            st.stop()

    # Backend connection check
    try:
        symbols = fetch_symbols()
    except requests.exceptions.Timeout as e:
        st.error(f"⏱️ **Backend Timeout**: {str(e)}")
        st.info("💡 **Troubleshooting**:")
        st.markdown("""
        1. Check if the backend is running: Look for `INFO: Uvicorn running on http://0.0.0.0:8000` in your terminal
        2. Wait a few seconds and click the button below to retry
        3. If the backend isn't running, start it with: `python app.py`
        """)
        if st.button("🔄 Retry Connection"):
            st.rerun()
        st.stop()
    except requests.exceptions.ConnectionError as e:
        st.error(f"🔌 **Connection Error**: {str(e)}")
        st.info("💡 **Troubleshooting**:")
        st.markdown("""
        1. Ensure the FastAPI backend is running on port 8000
        2. Start the backend with: `python app.py` (this starts both backend and frontend)
        3. Or run separately:
           - Backend: `uvicorn api.routes:app --host 0.0.0.0 --port 8000`
           - Frontend: `streamlit run frontend/dashboard.py --server.port 8501`
        """)
        if st.button("🔄 Retry Connection"):
            st.rerun()
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error**: Failed to fetch symbols from backend: {e}")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()

    cfg = layout_sidebar(symbols)
    selected = cfg["selected_symbols"]

    if len(selected) == 0:
        st.info("Select at least one symbol in the sidebar.")
        st.stop()

    col_main, col_side = st.columns([3, 1])

    # Handle CSV upload
    upload_cfg = cfg["upload"]
    if upload_cfg["run"] and upload_cfg["file"] is not None:
        try:
            res = upload_ohlc_csv(
                upload_cfg["file"],
                symbol=upload_cfg["symbol"],
                interval=upload_cfg["interval"],
            )
            st.success(f"Uploaded {res.get('rows', 0)} OHLC rows for {upload_cfg['symbol']}.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    # Optional simple auto-refresh using a rerun loop
    auto_refresh = st.sidebar.checkbox("Auto‑refresh (1s)", value=False)

    # Pick reference symbol for pair analytics (symbol_y vs symbol_x)
    symbol_y = selected[0]
    symbol_x = selected[1] if len(selected) > 1 else selected[0]

    interval = cfg["interval"]
    window = cfg["window"]
    regression = cfg["regression"]

    # Check data status
    data_status = fetch_data_status()
    ohlc_y_count = data_status.get("ohlc", {}).get(symbol_y, {}).get(interval, 0)
    ohlc_x_count = data_status.get("ohlc", {}).get(symbol_x, {}).get(interval, 0)
    
    if ohlc_y_count == 0 or ohlc_x_count == 0:
        st.warning(
            f"⚠️ **Data Status**: "
            f"{symbol_y} has {ohlc_y_count} {interval} bars, "
            f"{symbol_x} has {ohlc_x_count} {interval} bars. "
            f"Please wait for data ingestion to complete (usually 1-2 minutes). "
            f"The WebSocket is collecting ticks and resampling them into OHLC bars."
        )
        if st.button("Check Data Status Again"):
            st.rerun()
        st.stop()

    with st.spinner("Fetching analytics..."):
        try:
            run_adf = st.button("Run ADF test on spread")
            analytics = fetch_pair_analytics(
                symbol_y=symbol_y,
                symbol_x=symbol_x,
                interval=interval,
                window=window,
                regression=regression,
                run_adf=run_adf,
            )
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ **Failed to fetch analytics**: {error_msg}")
            if "Not enough" in error_msg or "No OHLC data" in error_msg:
                st.info("💡 **Tip**: Wait a few more seconds for data ingestion, or try a different interval (1s, 5m).")
            return

    # Price charts
    with col_main:
        st.subheader("Price & Volume")
        df_y = fetch_ohlc(symbol_y, interval)
        df_x = fetch_ohlc(symbol_x, interval)
        if df_y.empty or df_x.empty:
            st.warning("Not enough OHLC data yet. Wait a few seconds for ingestion/resampling.")
        else:
            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(
                    x=df_y.index,
                    y=df_y["close"],
                    mode="lines",
                    name=f"{symbol_y} close",
                )
            )
            if symbol_x != symbol_y:
                price_fig.add_trace(
                    go.Scatter(
                        x=df_x.index,
                        y=df_x["close"],
                        mode="lines",
                        name=f"{symbol_x} close",
                    )
                )
            price_fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(price_fig, use_container_width=True)

            vol_fig = px.bar(
                df_y.reset_index(),
                x="open_time",
                y="volume",
                title=f"{symbol_y} volume ({interval})",
            )
            vol_fig.update_layout(height=200, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(vol_fig, use_container_width=True)

        # Spread & z-score
        st.subheader("Spread & Z‑Score")
        spread_df = pd.DataFrame(analytics["spread"])
        z_df = pd.DataFrame(analytics["zscore"])
        if not spread_df.empty:
            spread_df["time"] = pd.to_datetime(spread_df["time"])
            z_df["time"] = pd.to_datetime(z_df["time"])

            spread_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=spread_df["time"],
                        y=spread_df["spread"],
                        mode="lines",
                        name="Spread",
                    )
                ]
            )
            spread_fig.update_layout(height=250, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(spread_fig, use_container_width=True)

            z_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=z_df["time"],
                        y=z_df["z"],
                        mode="lines",
                        name="Z‑score",
                    )
                ]
            )
            z_fig.add_hline(y=2.0, line_dash="dash", line_color="red")
            z_fig.add_hline(y=-2.0, line_dash="dash", line_color="green")
            z_fig.update_layout(height=250, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(z_fig, use_container_width=True)

        # Rolling correlation
        st.subheader("Rolling Correlation")
        corr_df = pd.DataFrame(analytics["correlation"])
        if not corr_df.empty:
            corr_df["time"] = pd.to_datetime(corr_df["time"])
            corr_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=corr_df["time"],
                        y=corr_df["corr"],
                        mode="lines",
                        name="Rolling corr",
                    )
                ]
            )
            corr_fig.update_layout(height=250, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(corr_fig, use_container_width=True)

    with col_side:
        st.subheader("Summary Stats")
        stats_y = analytics["stats"]["y"]
        stats_x = analytics["stats"]["x"]
        if stats_y and stats_x:
            stats_table = pd.DataFrame(
                {
                    "Metric": [
                        "Mean price",
                        "Std price",
                        "Last price",
                        "Last return",
                        "Ann. volatility",
                    ],
                    symbol_y: [
                        stats_y["mean_price"],
                        stats_y["std_price"],
                        stats_y["last_price"],
                        stats_y["last_return"],
                        stats_y["volatility_annualized"],
                    ],
                    symbol_x: [
                        stats_x["mean_price"],
                        stats_x["std_price"],
                        stats_x["last_price"],
                        stats_x["last_return"],
                        stats_x["volatility_annualized"],
                    ],
                }
            )
            st.table(stats_table)

        st.markdown("---")
        st.subheader("Hedge & ADF")
        st.metric("Hedge ratio", f"{analytics['hedge_ratio']:.4f}")
        if "adf" in analytics:
            adf = analytics["adf"]
            st.metric("ADF p‑value", f"{adf['pvalue']:.4f}")
            st.caption("Lower p‑values suggest a more stationary (mean‑reverting) spread.")

        st.markdown("---")
        st.subheader("Alerts")
        alert_cfg = cfg["alert"]
        if alert_cfg["create"]:
            rule = {
                "name": alert_cfg["name"],
                "symbol_y": symbol_y,
                "symbol_x": symbol_x,
                "metric": "zscore",
                "threshold": alert_cfg["threshold"],
                "direction": alert_cfg["direction"],
            }
            try:
                res = create_alert_api(rule)
                st.success(f"Created alert {res['rule']['name']}")
            except Exception as e:
                st.error(f"Failed to create alert: {e}")

        rules = list_alerts_api()
        if rules:
            st.write("Configured alerts:")
            st.json(rules)
        else:
            st.write("No alerts defined yet.")

        if analytics["alerts_triggered"]:
            st.error("Alerts triggered:")
            for a in analytics["alerts_triggered"]:
                st.write(f"- {a['message']} (z = {a['value']:.2f})")

        st.markdown("---")
        st.subheader("Backtest")
        bt_cfg = cfg["backtest"]
        if bt_cfg["run"]:
            try:
                bt = run_backtest_api(
                    symbol_y,
                    symbol_x,
                    interval,
                    window,
                    regression,
                    bt_cfg["entry_z"],
                    bt_cfg["exit_z"],
                )
                st.success(f"Backtest PnL: {bt['pnl']:.4f}")
                eq_df = pd.DataFrame(bt["equity_curve"])
                eq_df["time"] = pd.to_datetime(eq_df["time"])
                eq_fig = px.line(eq_df, x="time", y="equity", title="Equity curve")
                eq_fig.update_layout(height=250, margin=dict(l=40, r=20, t=30, b=40))
                st.plotly_chart(eq_fig, use_container_width=True)

                st.download_button(
                    "Download backtest CSV",
                    data=pd.DataFrame(bt["records"]).to_csv(index=False).encode("utf-8"),
                    file_name=f"backtest_{symbol_y}_{symbol_x}_{interval}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    # Trigger a rerun at the end if auto-refresh is enabled
    if auto_refresh:
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()



