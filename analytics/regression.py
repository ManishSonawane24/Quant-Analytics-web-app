from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT


RegressionType = Literal["ols", "robust", "kalman"]


@dataclass
class HedgeResult:
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    zscore: pd.Series


def _add_const(x: pd.Series) -> np.ndarray:
    return sm.add_constant(x.values)


def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """
    Classic OLS hedge ratio between two price series.

    y_t = alpha + beta * x_t + epsilon_t
    hedge_ratio = beta
    """
    x_const = _add_const(x)
    model = sm.OLS(y.values, x_const, missing="drop")
    res = model.fit()
    intercept = float(res.params[0])
    beta = float(res.params[1])
    return beta, intercept


def robust_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """
    Robust regression hedge ratio using Huber loss.

    This reduces the impact of outliers, which are common in high‑frequency
    data and can distort a simple OLS estimate.
    """
    x_const = _add_const(x)
    model = RLM(y.values, x_const, M=HuberT())
    res = model.fit()
    intercept = float(res.params[0])
    beta = float(res.params[1])
    return beta, intercept


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, q: float = 1e-5, r: float = 1e-3) -> pd.Series:
    """
    Dynamic hedge ratio using a simple 1‑state Kalman filter.

    State equation:
        beta_t = beta_{t-1} + w_t      (w_t ~ N(0, q))
    Observation:
        y_t = beta_t * x_t + v_t       (v_t ~ N(0, r))

    This allows the beta to evolve slowly over time, which can better
    capture regime changes than a static OLS estimate.
    """
    y_vals = y.values
    x_vals = x.values

    n = len(y_vals)
    if n == 0:
        return pd.Series(dtype=float)

    beta = np.zeros(n)
    P = 1.0  # Initial state variance

    for t in range(1, n):
        # Predict
        beta_pred = beta[t - 1]
        P_pred = P + q

        # Observation update
        H = x_vals[t]
        S = H * H * P_pred + r
        K = P_pred * H / S
        y_pred = beta_pred * H
        beta[t] = beta_pred + K * (y_vals[t] - y_pred)
        P = (1 - K * H) * P_pred

    return pd.Series(beta, index=y.index)


def build_spread_and_zscore(
    y: pd.Series,
    x: pd.Series,
    regression_type: RegressionType,
    window: int,
) -> HedgeResult:
    """
    Compute hedge ratio, spread and rolling z‑score for a pair.

    - y: dependent leg (e.g. BTCUSDT)
    - x: independent leg (e.g. ETHUSDT)
    - regression_type: 'ols', 'robust', or 'kalman'
    - window: rolling window for z‑score
    """
    y, x = y.align(x, join="inner")
    if y.empty or x.empty:
        raise ValueError("Price series must not be empty")

    if regression_type == "ols":
        beta, intercept = ols_hedge_ratio(y, x)
        spread = y - (intercept + beta * x)
    elif regression_type == "robust":
        beta, intercept = robust_hedge_ratio(y, x)
        spread = y - (intercept + beta * x)
    elif regression_type == "kalman":
        beta_series = kalman_hedge_ratio(y, x)
        intercept = 0.0
        spread = y - beta_series * x
        beta = float(beta_series.iloc[-1])
    else:
        raise ValueError(f"Unsupported regression type: {regression_type}")

    from .stats import rolling_zscore  # local import to avoid cycles

    z = rolling_zscore(spread, window=window)
    return HedgeResult(hedge_ratio=beta, intercept=intercept, spread=spread, zscore=z)



