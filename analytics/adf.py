from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    statistic: float
    pvalue: float
    lags_used: int
    nobs: int
    crit_values: Dict[str, float]
    icbest: Optional[float]


def run_adf(series: pd.Series, maxlag: Optional[int] = None) -> ADFResult:
    """
    Run Augmented Dickey‑Fuller test on a time series.

    This is a standard stationarity test; for a mean‑reverting spread,
    we generally want to see a small p‑value (reject unit root).
    """
    series = series.dropna().astype(float)
    if series.empty:
        raise ValueError("Series is empty for ADF test")

    result = adfuller(series.values, maxlag=maxlag, autolag="AIC")
    stat, pvalue, lags, nobs, crit_vals, icbest = result
    return ADFResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        lags_used=int(lags),
        nobs=int(nobs),
        crit_values={k: float(v) for k, v in crit_vals.items()},
        icbest=float(icbest) if icbest is not None else None,
    )



