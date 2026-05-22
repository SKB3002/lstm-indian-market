"""Exogenous feature engineering for ARMAX / GARCH-X models.

All features are constructed such that the value at timestamp ``t`` is
*observable at time t* — i.e. uses information up to and including bar t-1.
This is critical: any feature that leaks the current bar's information into
itself contaminates the walk-forward and inflates apparent performance.

Returned features:

    nifty_lag    : log-return of NIFTY 50 on bar t-1 (broad market direction)
    vix_lag      : log-level of India VIX on bar t-1 (market fear regime)
    vix_chg_lag  : log-change of India VIX on bar t-1 (regime *shift*)
    park_var_lag : Parkinson range estimator on bar t-1 — proxies realized
                   variance much less noisily than r^2:
                       parkinson = 0.361 * (log(high) - log(low))^2
    tod_sin, tod_cos : sin/cos of bar's position within the trading day
                       (encodes the U-shape: open + close vs midday)
    dow          : day-of-week 0..4 dummy (categorical, optional use)

Index of returned frame is the *stock's* timestamp index — yfinance ticks
should align across instruments at the same exchange but we left-join
defensively to avoid silent misalignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_bars


_BARS_PER_DAY_15M = 25  # NSE session has 25 fifteen-minute bars (09:15..15:15 inclusive)


def parkinson_variance(ohlc: pd.DataFrame) -> pd.Series:
    """0.361 * (log(high) - log(low))^2.

    The Parkinson 1980 estimator. Variance of log-returns assuming the bar's
    high and low fully bracket the random-walk path within the bar. Hugely
    less noisy than r^2 (which only sees close-to-close). For a stationary
    log Brownian motion within the bar, Parkinson reduces variance by ~5x vs
    the squared close-to-close return.
    """
    h = np.log(ohlc["high"].astype(float))
    l = np.log(ohlc["low"].astype(float))
    pv = 0.361 * (h - l) ** 2
    pv.name = "park_var"
    return pv


def time_of_day(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Sin/cos pair over the trading-day bar position.

    Maps 09:15 -> 0, 09:30 -> 1, ... 15:15 -> 24 then takes sin/cos with
    period 25. Lets a linear model capture the U-shape with two regressors.
    """
    minutes_since_open = (
        (index.hour - 9) * 60 + (index.minute - 15)
    )
    # Clip safely to handle a stray pre/post-session bar if any slipped through.
    bar_idx = np.clip(minutes_since_open // 15, 0, _BARS_PER_DAY_15M - 1)
    theta = 2.0 * np.pi * bar_idx / _BARS_PER_DAY_15M
    return (
        pd.Series(np.sin(theta), index=index, name="tod_sin"),
        pd.Series(np.cos(theta), index=index, name="tod_cos"),
    )


def build_features(
    stock_ohlc: pd.DataFrame,
    interval: str = "15m",
    period: str = "60d",
) -> pd.DataFrame:
    """Build the full exogenous feature matrix aligned to a stock's bar index.

    Pulls NIFTY and VIX bars (with the same interval/period) and aligns by
    timestamp. Missing values are forward-filled per series (small gaps) then
    rows with any remaining NaN are dropped.

    Args:
        stock_ohlc : DataFrame of the stock's OHLCV bars, IST tz-aware index.
        interval, period : passed to ``load_bars`` for fetching NIFTY + VIX.

    Returns:
        DataFrame indexed by stock_ohlc.index with columns:
            nifty_lag, vix_lag, vix_chg_lag, park_var_lag, tod_sin, tod_cos
        plus the integer 'bar_of_day' for diagnostics.
    """
    if not isinstance(stock_ohlc.index, pd.DatetimeIndex):
        raise TypeError("stock_ohlc must have a DatetimeIndex.")

    # 1) NIFTY return (lag-1)
    nifty = load_bars("NIFTY", interval=interval, period=period, use_cache=True)
    nifty_ret = np.log(nifty["close"].astype(float)).diff()
    nifty_lag = nifty_ret.shift(1).rename("nifty_lag")

    # 2) VIX level + change (lag-1)
    vix = load_bars("INDIAVIX", interval=interval, period=period, use_cache=True)
    vix_log = np.log(vix["close"].astype(float))
    vix_lag = vix_log.shift(1).rename("vix_lag")
    vix_chg_lag = vix_log.diff().shift(1).rename("vix_chg_lag")

    # 3) Parkinson realized-variance on the stock (lag-1)
    park = parkinson_variance(stock_ohlc).shift(1).rename("park_var_lag")

    # 4) Time-of-day (contemporaneous — deterministic, no leakage)
    tod_sin, tod_cos = time_of_day(stock_ohlc.index)

    # Assemble — left-join on stock_ohlc index keeps the stock as primary clock.
    feats = pd.concat(
        [nifty_lag, vix_lag, vix_chg_lag, park, tod_sin, tod_cos],
        axis=1,
    ).reindex(stock_ohlc.index)

    # Forward-fill small gaps (one missing index point), then drop residual NaNs.
    feats = feats.ffill(limit=2).dropna()
    return feats
