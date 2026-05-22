"""Stationarity, autocorrelation, and residual diagnostics.

Thin, opinionated wrappers around statsmodels. The goal is one-line calls in
notebooks that print a verdict you can read at a glance, not just raw numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityVerdict:
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float

    @property
    def adf_stationary(self) -> bool:
        # ADF H0 = unit root (non-stationary). Reject => stationary.
        return self.adf_pvalue < 0.05

    @property
    def kpss_stationary(self) -> bool:
        # KPSS H0 = stationary. Fail to reject => stationary.
        return self.kpss_pvalue > 0.05

    @property
    def verdict(self) -> str:
        a, k = self.adf_stationary, self.kpss_stationary
        if a and k:
            return "stationary (ADF + KPSS agree)"
        if not a and not k:
            return "non-stationary (ADF + KPSS agree)"
        if a and not k:
            return "trend-stationary or difference-stationary (tests disagree)"
        return "borderline; consider differencing"

    def __repr__(self) -> str:
        return (
            f"ADF stat={self.adf_stat:.3f} p={self.adf_pvalue:.4f} | "
            f"KPSS stat={self.kpss_stat:.3f} p={self.kpss_pvalue:.4f} | "
            f"{self.verdict}"
        )


def check_stationarity(series: pd.Series, regression: str = "c") -> StationarityVerdict:
    """Run ADF + KPSS jointly. Drops NaNs internally.

    regression: 'c' = constant, 'ct' = constant + trend. Use 'c' for returns,
    'ct' for raw price series.
    """
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, regression=regression, autolag="AIC")
    # statsmodels emits a noisy InterpolationWarning when p is outside the
    # tabulated range; we don't care for a verdict.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(s, regression=regression, nlags="auto")
    return StationarityVerdict(adf_stat, adf_p, kpss_stat, kpss_p)


def ljung_box(resid: pd.Series, lags: int = 20) -> pd.DataFrame:
    """Ljung-Box test for residual autocorrelation. Returns the statsmodels table."""
    return acorr_ljungbox(resid.dropna(), lags=lags, return_df=True)


def normality(resid: pd.Series) -> dict:
    """Jarque-Bera. H0 = normal. Heavy-tailed financial returns will reject hard."""
    jb, p, skew, kurt = jarque_bera(resid.dropna())
    return {"jb_stat": jb, "pvalue": p, "skew": skew, "excess_kurtosis": kurt - 3}


def intraday_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Average |return| by time-of-day. Surfaces the U-shape clearly."""
    df = pd.DataFrame({"abs_ret": returns.abs()})
    df["tod"] = returns.index.strftime("%H:%M")
    return df.groupby("tod")["abs_ret"].agg(["mean", "std", "count"])
