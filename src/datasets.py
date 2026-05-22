"""Sequence dataset builder for LSTM models on intraday bars.

The whole reason this module exists is to make look-ahead leakage *structurally
impossible* rather than a thing you have to remember to avoid. Two rules:

1. The target at row t is built from information strictly after t (e.g. the
   next bar's log-return). The features at row t are built from information up
   to and including t. Splitting features and target into separate frames at
   build time, then aligning by index, keeps that invariant verifiable.
2. The train/val/test split is purely chronological. No shuffling across the
   time axis. The scaler is fit on the training slice only — applying a
   scaler fit on the full series silently leaks future moments into training.

Sliding windows are produced as contiguous tensor blocks: X has shape
(n_windows, seq_len, n_features), y has shape (n_windows,) for scalar targets
or (n_windows, k) for k-step ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def next_bar_logret(close: pd.Series) -> pd.Series:
    """Target: log-return of the *next* bar.

    target[t] = log(close[t+1] / close[t])

    The last observation has no next bar, so it ends up NaN and gets dropped.
    """
    return np.log(close.shift(-1) / close).rename("y_next_logret")


def next_k_bar_logret(close: pd.Series, k: int) -> pd.Series:
    """Target: cumulative log-return over the next k bars.

    target[t] = log(close[t+k] / close[t]) = sum of next k 1-bar log-returns.
    Used for the longer-horizon experiments (Phase 3).
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    return np.log(close.shift(-k) / close).rename(f"y_next_{k}bar_logret")


def as_percent_return(logret: pd.Series) -> pd.Series:
    """Convert log-returns to percent (multiply by 100).

    Why: training a regression head on raw log-returns means targets live near
    ~1e-3 while inputs are O(1) after scaling. The final linear layer has to
    learn a ~1000x shrink, which makes losses uninterpretably small and slows
    early training. Multiplying by 100 puts targets in a human-readable unit
    (percent) with no risk of leakage — it's a fixed constant, not a fitted
    statistic. Undo at inference with ``pred / 100.0``.
    """
    return (logret * 100.0).rename(f"{logret.name}_pct" if logret.name else "y_pct")


def next_bar_direction(close: pd.Series) -> pd.Series:
    """Target: sign of next bar log-return. 0 -> down, 1 -> up. Zero -> NaN.

    Strictly binary; ties (exact zero next return) are dropped rather than
    arbitrarily assigned, since dropping ~handful of bars beats biasing the
    classifier.
    """
    nxt = np.log(close.shift(-1) / close)
    direction = pd.Series(np.where(nxt > 0, 1.0, np.where(nxt < 0, 0.0, np.nan)),
                          index=close.index, name="y_next_dir")
    return direction


# ---------------------------------------------------------------------------
# Standardization (no-leak)
# ---------------------------------------------------------------------------


@dataclass
class StandardScaler:
    """Per-column mean/std. Fit on training rows only; apply elsewhere.

    Deliberately written as a tiny class rather than reaching for sklearn:
    it makes the no-leak contract obvious and keeps the dependency surface
    minimal.
    """

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        # Guard against zero-variance columns (e.g. constant time-of-day in a
        # tiny slice). Replace zero std with 1.0; the column becomes a constant
        # zero feature, which is harmless.
        self.std_ = np.where(self.std_ > 1e-12, self.std_, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fit.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


@dataclass
class SplitIndex:
    """Row-index boundaries for train/val/test slices of a feature frame.

    All three slices are contiguous and non-overlapping. The convention is:
        train  = [0, train_end)
        val    = [train_end, val_end)
        test   = [val_end, n)
    """

    train_end: int
    val_end: int
    n: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.val_end, self.n)


def chronological_split(
    n: int,
    test_bars: int,
    val_frac_of_train: float = 0.2,
) -> SplitIndex:
    """Carve a row-aligned frame into train / val / test.

    Args:
        n: total number of rows in the aligned features+target frame.
        test_bars: how many of the most-recent bars to hold out for paper trade.
            For the 12-day window on 15m bars: 12 * 25 = 300.
        val_frac_of_train: fraction of the training slice (the leftover after
            test is reserved) to use as validation, taken from its *tail* so
            early-stopping sees the freshest pre-test bars.

    Returns:
        A SplitIndex with computed boundaries.
    """
    if test_bars >= n:
        raise ValueError(f"test_bars={test_bars} >= n={n}.")
    pre_test = n - test_bars
    val_bars = int(round(pre_test * val_frac_of_train))
    train_end = pre_test - val_bars
    val_end = pre_test
    if train_end < 1:
        raise ValueError("Training slice ended up empty — n too small or val_frac too high.")
    return SplitIndex(train_end=train_end, val_end=val_end, n=n)


# ---------------------------------------------------------------------------
# Sliding window construction
# ---------------------------------------------------------------------------


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (n_windows, seq_len, n_feat), (n_windows,), end-row-index arrays.

    Window i covers rows [i, i+seq_len) of X, predicting y[i+seq_len-1].
    That means the LSTM "sees" rows i..i+seq_len-1 and emits the target at
    the last seen row — which is itself a target built from STRICTLY future
    information (next bar) at that timestamp. No leakage.

    Returns:
        Xs: float32 array, shape (W, seq_len, n_feat)
        ys: float32 array, shape (W,)
        end_idx: int array of length W, where end_idx[i] is the row index in
                 the original frame at which window i's prediction lives
                 (= i + seq_len - 1). Useful for re-aligning predictions to
                 the source DatetimeIndex.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_rows, n_features).")
    if len(X) != len(y):
        raise ValueError("X and y must have equal length.")
    n = len(X)
    W = n - seq_len + 1
    if W <= 0:
        raise ValueError(f"seq_len={seq_len} >= n={n}; no windows produced.")

    n_feat = X.shape[1]
    Xs = np.empty((W, seq_len, n_feat), dtype=np.float32)
    for i in range(W):
        Xs[i] = X[i : i + seq_len]
    end_idx = np.arange(seq_len - 1, n)
    ys = y[end_idx].astype(np.float32)
    return Xs, ys, end_idx


# ---------------------------------------------------------------------------
# One-shot builder used by Phase 1 / Phase 2 notebooks
# ---------------------------------------------------------------------------


@dataclass
class SequenceBundle:
    """All tensors a training loop needs, plus index metadata to map back."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_ts: pd.DatetimeIndex
    scaler: StandardScaler
    feature_names: list[str]
    target_name: str
    target_unit: str  # "percent", "logret", or "binary" — required for honest backtest scaling

    @property
    def n_features(self) -> int:
        return self.X_train.shape[2]

    @property
    def seq_len(self) -> int:
        return self.X_train.shape[1]


def build_sequence_bundle(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int,
    test_bars: int,
    target_unit: str,
    val_frac_of_train: float = 0.2,
) -> SequenceBundle:
    """End-to-end: align frames, split, scale (train-only fit), window.

    Args:
        features: feature DataFrame, DatetimeIndex aligned with `target`.
        target: target Series. Must be observable-from-the-future already
            (i.e. produced by next_bar_logret / next_bar_direction). Rows
            with NaN in either frame are dropped before splitting.
        seq_len: LSTM context length in bars.
        test_bars: number of most-recent bars to hold out for paper trade.
        target_unit: one of "percent", "logret", "binary". Travels with the
            bundle so downstream backtest code knows how to interpret
            predictions (e.g. divide by 100 before feeding the cost model).
        val_frac_of_train: fraction of the pre-test slice used as validation.

    Returns:
        SequenceBundle with windowed train/val/test tensors and metadata.
    """
    # 1. Align and drop NaNs.
    df = features.join(target, how="inner").dropna()
    feat_cols = list(features.columns)
    tgt_col = target.name
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df[tgt_col].to_numpy(dtype=np.float64)
    idx = df.index

    # 2. Chronological split.
    sp = chronological_split(len(df), test_bars=test_bars,
                             val_frac_of_train=val_frac_of_train)

    # 3. Fit scaler on training rows ONLY, then transform all slices.
    scaler = StandardScaler().fit(X[sp.train_slice])
    X_scaled = scaler.transform(X)

    # 4. Window each split independently so windows can't straddle boundaries.
    Xt, yt, _ = make_sequences(X_scaled[sp.train_slice], y[sp.train_slice], seq_len)
    Xv, yv, _ = make_sequences(X_scaled[sp.val_slice], y[sp.val_slice], seq_len)
    Xe, ye, end_idx_test = make_sequences(X_scaled[sp.test_slice], y[sp.test_slice], seq_len)

    # Map test windows back to source timestamps for blotter alignment.
    test_ts = idx[sp.test_slice][end_idx_test]

    return SequenceBundle(
        X_train=Xt, y_train=yt,
        X_val=Xv, y_val=yv,
        X_test=Xe, y_test=ye,
        test_ts=test_ts,
        scaler=scaler,
        feature_names=feat_cols,
        target_name=str(tgt_col),
        target_unit=target_unit,
    )
