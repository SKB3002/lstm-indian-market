"""Data layer for Indian intraday equity bars.

Pulls OHLCV from yfinance, normalizes timezone to Asia/Kolkata, filters to the
NSE cash session (09:15–15:30 IST), and caches as parquet for fast reload.

Free yfinance limits to be aware of:
    interval='1m'   ~7 days history
    interval='5m'   ~60 days history
    interval='15m'  ~60 days history
    interval='1d'   ~decades

For multi-year intraday history a paid feed (Kite/Dhan/GDFL) is required; that
swap-out lives behind the same load_bars() signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf

IST: Final = "Asia/Kolkata"
SESSION_START: Final = "09:15"
SESSION_END: Final = "15:30"

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
RAW_DIR: Final = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final = PROJECT_ROOT / "data" / "processed"

# Common Indian tickers on Yahoo Finance
TICKERS: Final = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "INDIAVIX": "^INDIAVIX",
}


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str, interval: str, period: str) -> Path:
    safe = symbol.replace("^", "").replace(".", "_")
    return PROCESSED_DIR / f"{safe}_{interval}_{period}.parquet"


def _to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Force the DatetimeIndex into Asia/Kolkata. Idempotent."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert(IST)
    return df


def _filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars whose timestamp falls inside the NSE cash session.

    yfinance bars are *open-time* stamped, so a bar labeled 15:15 covers
    15:15–15:30 and is the last legitimate bar of the day.
    """
    return df.between_time(SESSION_START, SESSION_END, inclusive="left")


def load_bars(
    symbol: str = "NIFTY",
    interval: str = "15m",
    period: str = "60d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV bars for a given symbol, cached as parquet.

    Args:
        symbol: Either a key in TICKERS or a raw Yahoo ticker.
        interval: yfinance interval string ('1m','5m','15m','30m','1h','1d').
        period: yfinance period string ('7d','60d','1y','max').
        use_cache: When True, returns cached parquet if present.

    Returns:
        DataFrame with columns [open, high, low, close, volume] indexed by
        timezone-aware (Asia/Kolkata) DatetimeIndex, session-filtered.
    """
    _ensure_dirs()
    yahoo_ticker = TICKERS.get(symbol, symbol)
    cache = _cache_path(yahoo_ticker, interval, period)

    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        return df

    raw = yf.Ticker(yahoo_ticker).history(
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
    )
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {yahoo_ticker} "
            f"(interval={interval}, period={period})"
        )

    # Normalize columns: yfinance gives Title Case; lower-case for cleanliness.
    raw.columns = [c.lower() for c in raw.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
    df = raw[keep].copy()

    df = _to_ist(df)
    # Intraday intervals only: filter to session. Daily bars skip this.
    if interval.endswith(("m", "h")):
        df = _filter_session(df)

    df = df.sort_index()
    # Drop fully-NaN rows that sometimes appear at session boundaries
    df = df.dropna(how="all")

    df.to_parquet(cache)
    return df


def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Append simple return and log-return columns. Drops the first NaN row."""
    import numpy as np

    out = df.copy()
    out["ret"] = out[price_col].pct_change()
    out["logret"] = (out[price_col] / out[price_col].shift(1)).map(np.log)
    return out.dropna(subset=["logret"])


def session_groups(df: pd.DataFrame) -> pd.api.typing.DataFrameGroupBy:
    """Group bars by trading date (useful when you must avoid overnight gaps)."""
    return df.groupby(df.index.normalize())
