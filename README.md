# Quant Analytics Web App – Real‑Time Binance Futures Dashboard

## Overview

This project is a **real‑time analytical trading dashboard** designed for quantitative trading and stat‑arb / mean‑reversion research. It:

- **Ingests live Binance Futures trade ticks via WebSocket**
- **Stores raw ticks and resampled OHLCV bars (1s / 1m / 5m) in SQLite**
- **Computes core quantitative analytics** (price stats, hedge ratios, spreads, z‑scores, ADF tests, rolling correlations)
- **Implements advanced analytics** (Kalman filter dynamic hedge ratio, robust regression, mean‑reversion backtest)
- **Exposes a FastAPI backend (REST) and a Streamlit + Plotly dashboard frontend**
- **Supports alerts, CSV upload, and CSV export for OHLC and backtests**

The entire system is designed as a **proto‑production architecture** with modular packages (`ingestion/`, `storage/`, `analytics/`, `alerts/`, `api/`, `frontend/`) and can be extended to additional exchanges, symbols, or analytics.

You can run the complete stack (backend + frontend) locally with:

```bash
python app.py
```

---

## 1. Project Structure

```text
.
├── app.py                  # Main entry point – starts FastAPI (uvicorn) + Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This document
├── data/
│   └── trading.db          # SQLite DB (created at runtime)
├── ingestion/
│   ├── __init__.py
│   └── websocket_client.py # Binance Futures WebSocket client (symbol@trade → ticks)
├── storage/
│   ├── __init__.py
│   └── db.py               # SQLite persistence for ticks, OHLC, backtest metadata
├── analytics/
│   ├── __init__.py
│   ├── resampling.py       # Tick → OHLCV resampling (1s, 1m, 5m)
│   ├── stats.py            # Price stats, returns, z‑scores, correlations, backtests
│   ├── regression.py       # OLS / robust / Kalman hedge ratios, spreads, z‑scores
│   └── adf.py              # Augmented Dickey‑Fuller stationarity test
├── alerts/
│   ├── __init__.py
│   └── rules.py            # User‑defined alert rules (e.g. z‑score > 2)
├── api/
│   ├── __init__.py
│   └── routes.py           # FastAPI app: REST endpoints, background tasks, CSV export
└── frontend/
    ├── __init__.py
    └── dashboard.py        # Streamlit + Plotly dashboard for interactive analytics
```

---

## 2. Setup & Running Locally

### 2.1. Prerequisites

- **Python**: 3.10+ (tested on 3.10–3.12)
- **pip**: latest recommended
- Internet access to **Binance Futures WebSocket** endpoint:
  - `wss://fstream.binance.com/stream`

### 2.2. Install Dependencies

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- **Backend**: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`, `statsmodels`, `duckdb`, `redis` (optional), `websockets`
- **Frontend**: `streamlit`, `plotly`, `requests`

### 2.3. Run the Application

From the project root:

```bash
python app.py
```

This will:

1. Start the **FastAPI backend** with uvicorn on `http://localhost:8000`
2. Start the **Streamlit dashboard** on `http://localhost:8501`
3. Open your default browser to the Streamlit UI (depending on local settings)

If you prefer to run components separately:

```bash
# Terminal 1 – backend
uvicorn api.routes:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 – frontend
streamlit run frontend/dashboard.py --server.port 8501
```

---

## 3. Architecture & Data Flow

### 3.1. High‑Level Flow

1. **Ingestion (`ingestion/websocket_client.py`)**
   - Uses the official **Binance Futures WebSocket trade stream**:
     - Endpoint: `wss://fstream.binance.com/stream?streams=btcusdt@trade/ethusdt@trade/...`
     - Parses each `@trade` event:
       ```json
       {
         "e": "trade",
         "E": 123456789,
         "s": "BTCUSDT",
         "t": 12345,
         "p": "0.001",
         "q": "100",
         "b": 88,
         "a": 50,
         "T": 123456785,
         "m": true,
         "M": true
       }
       ```
   - Normalizes to `{timestamp, symbol, price, size}` using:
     - `timestamp` = `T` (trade time, ms since epoch)
     - `symbol` = `s` (e.g. `BTCUSDT`)
     - `price` = `float(p)`
     - `size` = `float(q)`
   - Feeds normalized `Trade` objects into the storage layer.

2. **Storage (`storage/db.py`)**
   - Uses **SQLite** (file `data/trading.db`) via the built‑in `sqlite3` module.
   - Tables:
     - `ticks` – raw trades
       - `ts_ms`, `symbol`, `price`, `size`
     - `ohlc` – resampled OHLCV bars
       - `symbol`, `interval` (`1s`, `1m`, `5m`), `open_time_ms`, `open`, `high`, `low`, `close`, `volume`, `source` (`live`/`upload`)
     - `backtests` – placeholder for persisting backtest metadata (extendable)
   - Uses **short‑lived connections** per operation to avoid threading issues.

3. **Resampling & Core Analytics (`analytics/`)**
   - `resampling.py`:
     - Converts recent `ticks` to **time‑indexed pandas DataFrames**.
     - Uses `DataFrame.resample()` to compute OHLCV:
       - `1s` → `1S`, `1m` → `1T`, `5m` → `5T`
       - Computes **open, high, low, close** on `price`, and **volume** as sum of `size`.
     - Upserts results into `ohlc` with `(symbol, interval, open_time_ms, source)` as a unique key.
   - `stats.py`:
     - `compute_returns`: log returns \(r_t = \ln(P_t) - \ln(P_{t-1})\)
     - `compute_price_stats`:
       - Mean price, std dev, last price, last return, annualized volatility
     - `rolling_zscore`:
       - \( z_t = \frac{x_t - \mu_t}{\sigma_t} \) over a rolling window
     - `rolling_correlation`:
       - Rolling Pearson correlation between two close series
     - `mean_reversion_backtest` (advanced analytic #1):
       - Rule: enter when \( |z| > z_{\text{entry}} \), exit when \( |z| < z_{\text{exit}} \)
       - Tracks positions, spread changes, PnL and cumulative equity.
   - `regression.py`:
     - **OLS hedge ratio**:
       - \( y_t = \alpha + \beta x_t + \epsilon_t \), hedge ratio = \(\beta\)
     - **Robust regression (Huber RLM)** (advanced analytic #2):
       - Reduces influence of outliers in high‑frequency data.
     - **Kalman filter hedge ratio** (advanced analytic #3):
       - Dynamic \(\beta_t\) updated via a simple 1‑state Kalman filter:
         - \( \beta_t = \beta_{t-1} + w_t \)
         - \( y_t = \beta_t x_t + v_t \)
       - Produces a **time‑varying hedge ratio** and spread series.
     - `build_spread_and_zscore`:
       - Aligns two price series, estimates hedge ratio (OLS / robust / Kalman),
       - Computes **spread** and **rolling z‑score**.
   - `adf.py`:
     - Wraps **statsmodels** `adfuller` for Augmented Dickey‑Fuller test.
     - Returns statistic, p‑value, lags, critical values, and IC.

4. **Alerts (`alerts/rules.py`)**
   - Contains an in‑memory `AlertRegistry` (thread‑safe) with:
     - `AlertRule`: describes a user‑defined alert, e.g. *"BTCUSDT/ETHUSDT z‑score > 2"*
     - `TriggeredAlert`: runtime events when conditions are met.
   - Currently supports:
     - Metric: `zscore`
     - Directions: `above` / `below`
   - The registry is used by the analytics API to evaluate alerts against the **latest z‑score** for each pair.
   - The design makes it straightforward to:
     - Back it with **Redis** for distributed alerting,
     - Add new metrics (spread, correlation, volume, etc.).

5. **API Layer (`api/routes.py`)**
   - A **FastAPI** application that:
     - On startup:
       - Calls `init_db()` to create tables.
       - Starts background asyncio tasks:
         - `BinanceFuturesClient.run_forever()` – live **WebSocket tick ingestion**.
         - `resample_loop()` – periodic resampling into `1s`, `1m`, `5m` OHLCV bars.
     - Exposes REST endpoints:
       - `GET /health` – health check.
       - `GET /symbols` – supported symbols (e.g. `["BTCUSDT", "ETHUSDT"]`).
       - `GET /ohlc` – fetch OHLCV bars for a symbol/interval/source.
       - `GET /analytics/pair` – core and advanced analytics:
         - Hedge ratio (OLS / robust / Kalman),
         - Spread series,
         - Rolling z‑score,
         - Rolling correlation,
         - Price statistics,
         - Optional **ADF test** on the spread,
         - Evaluated **alerts** for z‑score.
       - `POST /analytics/backtest` – mean‑reversion backtest on the spread.
       - `POST /alerts` / `GET /alerts` / `DELETE /alerts/{id}` – create/list/delete alert rules.
       - `POST /upload/ohlc` – ingest **OHLC CSV** into the same `ohlc` table (source=`upload`).
       - `GET /export/ohlc` – CSV download of OHLCV data.
       - `POST /export/backtest` – CSV download for backtest records.

6. **Frontend (`frontend/dashboard.py`)**
   - **Streamlit** dashboard connecting to the FastAPI backend via **HTTP (requests)**.
   - Sidebar controls:
     - **Symbol selector (multi‑select)** – pick 1–N symbols; first two form the pair (Y vs X).
     - **Timeframe selector** – `1s`, `1m`, `5m`.
     - **Rolling window input** – for z‑score and rolling correlation.
     - **Regression type selector** – `OLS`, `Robust (Huber)`, `Kalman`.
     - **Alert configuration** – name, z‑score threshold, direction; creates `/alerts` rules.
     - **Backtest configuration** – entry/exit z; runs `/analytics/backtest`.
     - **OHLC CSV upload** – posts to `/upload/ohlc` and reuses the same analytics pipeline.
   - Main panel:
     - **Price chart** (multi‑symbol, zoom/pan/hover via Plotly).
     - **Volume bars** for the primary symbol.
     - **Spread chart** and **z‑score chart** with ±2σ guide lines.
     - **Rolling correlation chart** between the two series.
     - **Summary stats table** for both legs (mean, std, last price, last return, annualized vol).
     - **Hedge ratio + ADF metrics** (p‑value).
     - **Alerts section** showing configured rules and any triggered alerts.
     - **Backtest section** with equity curve plot and CSV download of backtest records.

### 3.2. Live Update Strategy

- **Tick‑level ingestion** from Binance runs continuously via **async WebSocket**.
- **Resampled OHLCV**:
  - Background task updates `1s`, `1m`, `5m` bars every second.
  - Each interval’s bar is only finalized/resampled once its window is complete.
- **Frontend refresh**:
  - Uses periodic **REST polling** (via `requests`) instead of a browser WebSocket.
  - Plotly charts support **zoom, pan, hover** out of the box.

---

## 4. Analytics Details & Business Intuition

### 4.1. Price Statistics

- **Mean & Std Dev**:
  - Provide a quick sense of the central tendency and dispersion of prices over the selected window.
- **Last Return & Volatility**:
  - Log returns \( r_t = \ln(P_t) - \ln(P_{t-1}) \).
  - Annualized volatility derived from return standard deviation, assuming a specified number of bars per year.
  - Useful for risk budgeting and comparing assets with different price levels.

### 4.2. Pair Trading & Hedge Ratios

- **OLS Hedge Ratio**:
  - Models \( Y_t = \alpha + \beta X_t + \epsilon_t \).
  - \(\beta\) is the **hedge ratio**, giving the relative sizing of the two legs so that the residual (spread) represents the mispricing.
- **Robust Regression (Huber)**:
  - Down‑weights extreme moves that can distort OLS.
  - Particularly useful in crypto markets where large prints or fat‑finger trades are common.
- **Kalman Filter Hedge Ratio**:
  - Allows \(\beta_t\) to **evolve over time**, capturing structural changes in the relationship between assets.
  - Beneficial for **regime shifts** (e.g. changing correlations between BTC and ETH).

### 4.3. Spread, Z‑Score, and Mean‑Reversion

- **Spread**:
  - Defined as the residual of the regression (static or dynamic):
    - \( \text{spread}_t = Y_t - (\alpha + \beta X_t) \) (OLS/robust)
    - Or \( \text{spread}_t = Y_t - \beta_t X_t \) (Kalman)
  - A stationary spread is a good candidate for **mean‑reversion strategies**.
- **Z‑Score**:
  - Rolling normalization of the spread:
    - \( z_t = \frac{\text{spread}_t - \mu_t}{\sigma_t} \)
  - Values like \( |z_t| > 2 \) often indicate **statistically significant deviations**.
  - Used in alerting (e.g. “z‑score > 2”).

### 4.4. ADF Stationarity Test

- **Augmented Dickey‑Fuller (ADF)**:
  - Tests the null hypothesis that the series has a unit root (non‑stationary).
  - A **low p‑value** (e.g. < 0.05) supports rejecting the null, suggesting the spread is stationary.
  - For mean‑reversion strategies, a stationary spread is desirable (consistent pullback to a long‑run mean).

### 4.5. Rolling Correlation

- Measures the stability of the relationship between the two series over the chosen window.
- High and stable correlations are typically required for **pairs trading**:
  - If correlation collapses, the pair may be **decoupling**, and existing models/hedges may no longer be valid.

### 4.6. Mean‑Reversion Backtest

- Implements a simple **threshold‑based strategy**:
  - Enter when spread z‑score exceeds a high threshold \(z_{\text{entry}}\) in either direction.
  - Exit when the z‑score reverts toward zero (below \(z_{\text{exit}}\)).
- The backtest returns:
  - A **position time series**, trade‑by‑trade **PnL**, and **equity curve**.
  - CSV export for further analysis (e.g. slippage modeling, risk metrics, parameter sweeps).

---

## 5. Scaling & Production Considerations

This project is intentionally designed as a **proto‑production** architecture. To scale to real‑world trading or research loads:

1. **Database & Storage**
   - Migrate from SQLite to **PostgreSQL** or **DuckDB** for larger datasets and concurrent writes.
   - Introduce **partitioning** by symbol and date.
   - Offload raw tick storage to object storage (e.g. S3) with parquet files for historical backfill.

2. **Ingestion & Resilience**
   - Use **multiple ingestion workers** and a message queue (e.g. Kafka, Redis Streams).
   - Add robust **logging, metrics, and alerting** around WebSocket disconnects and API throttling.
   - Implement **backoff with jitter** and per‑symbol circuit breakers.

3. **Analytics & Latency**
   - Move heavy resampling and analytics to a **separate worker tier**.
   - Cache hot time series and analytics in **Redis** to avoid re‑computing for each request.
   - Use **vectorized** or **Numba‑accelerated** kernels for more complex analytics.

4. **Frontend & API**
   - Switch from Streamlit to a fully custom React/Next.js UI if you need tighter control over UX.
   - Add **authentication** and **role‑based access** for production use.
   - Introduce **rate limiting** and **pagination** for large exports.

5. **Alerting**
   - Persist alert rules and states in Redis or a database.
   - Add **push notifications** (Slack, email, SMS) and integrate with on‑call systems.
   - Implement **backtesting of alert rules** and **alert fatigue** controls.

---

## 6. Architecture Diagram Instructions (draw.io)

To create an architecture diagram in **draw.io (diagrams.net)**, you can use the following component list and connections:

### 6.1. Components (Shapes)

Create the following nodes:

1. **User Browser**
   - Connects to `Streamlit Dashboard` via HTTP (port 8501).
2. **Streamlit Dashboard (frontend/dashboard.py)**
   - Type: Web UI / Application.
3. **FastAPI Backend (api/routes.py)** running on `uvicorn`
   - Type: Application / Service.
4. **Binance Futures WebSocket (External Service)**
   - Endpoint: `wss://fstream.binance.com/stream`.
5. **Ingestion Service (ingestion/websocket_client.py)**
   - Sub‑component of FastAPI runtime (background task).
6. **Resampling & Analytics (analytics/*)**
   - Group shape containing:
     - `resampling.py`
     - `stats.py`
     - `regression.py`
     - `adf.py`
7. **Alert Registry (alerts/rules.py)**
   - Type: In‑memory store (or cache).
8. **SQLite Database (storage/db.py, data/trading.db)**
   - Type: Database.

### 6.2. Data Flow (Arrows)

Draw arrows as follows:

1. **User Browser → Streamlit Dashboard**
   - Label: `HTTP (port 8501)`
2. **Streamlit Dashboard → FastAPI Backend**
   - Label: `REST (JSON) via requests`
   - Endpoints: `/ohlc`, `/analytics/pair`, `/analytics/backtest`, `/alerts`, `/upload/ohlc`, `/export/*`
3. **FastAPI Backend → Ingestion Service**
   - Label: `Background tasks (asyncio create_task)`
4. **Ingestion Service → Binance Futures WebSocket**
   - Label: `WebSocket (symbol@trade)`
5. **Binance Futures WebSocket → Ingestion Service**
   - Label: `Trade ticks (JSON)`
6. **Ingestion Service → SQLite Database**
   - Label: `INSERT ticks (ts_ms, symbol, price, size)`
7. **Resampling & Analytics → SQLite Database**
   - Label: `READ ticks / ohlc; UPSERT ohlc`
8. **FastAPI Backend ↔ Resampling & Analytics**
   - Label: `Function calls (compute OHLC, stats, regression, ADF, backtest)`
9. **FastAPI Backend ↔ Alert Registry**
   - Label: `Create/list/delete/evaluate alerts`
10. **FastAPI Backend → Streamlit Dashboard**
    - Label: `Analytics responses (JSON)`

Arrange the diagram so that:

- **Top‑left**: User Browser.
- **Top‑center**: Streamlit Dashboard.
- **Center**: FastAPI Backend box containing:
  - Ingestion Service,
  - Resampling & Analytics,
  - Alert Registry.
- **Right**: Binance WebSocket (cloud icon).
- **Bottom‑center**: SQLite Database.

---

## 7. ChatGPT Usage Transparency

This project was **generated with the assistance of an AI language model (ChatGPT)** based on a detailed specification for a real‑time quantitative trading dashboard. The AI:

- Helped design the **modular architecture** (ingestion, storage, analytics, alerts, API, frontend).
- Produced initial implementations for:
  - Binance WebSocket ingestion,
  - SQLite storage and resampling logic,
  - Statistical and econometric analytics (OLS, robust/Kalman regression, ADF),
  - FastAPI routes and Streamlit dashboard.
- Wrote this README and the architecture description.

However:

- The generated code is intended as a **starting point / prototype**.
- Before using this system in **production or live trading**, you should:
  - **Review and test** all code paths, especially error handling and reconnection logic.
  - **Validate analytics** (hedge ratios, z‑scores, ADF, backtests) against your own research stack.
  - Add appropriate **risk controls, monitoring, logging, and security hardening**.

By using this project, you acknowledge that it is provided **as‑is** and should not be used for actual trading decisions without thorough independent verification.
