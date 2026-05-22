"""Strategy backtester with realistic NSE cost model.

We deliberately do not model the full microstructure (order book, partial fills,
auctions). What we *do* model honestly:

- Per-trade round-trip costs in basis points, configurable for intraday vs
  delivery-style settlement
- Slippage as a per-leg bps charge
- Costs proportional to the size of position change (so a small rebalance pays
  proportional cost, not a full round-trip)

Cost defaults are conservative for NIFTY-futures-like execution on Zerodha-tier
brokerage. Easy to override per backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NSECostModel:
    """Per-leg cost model. All values in bps (basis points = 0.01%).

    Defaults below approximate Zerodha-tier intraday NIFTY futures execution:
    - STT 1 bps on sell (intraday futures)
    - Brokerage 3 bps per leg (capped; ~ ₹20 per executed order or 0.03%)
    - Exchange + GST + SEBI + stamp combined ~ 0.5 bps per leg
    - Slippage 5 bps per leg on liquid index futures
    Total per leg ≈ 9 bps; round-trip ≈ 18 bps. We default a tad lower for
    futures and bump it for delivery in the from_preset classmethod.
    """

    brokerage_bps_per_leg: float = 3.0
    exchange_etc_bps_per_leg: float = 0.5
    slippage_bps_per_leg: float = 5.0
    stt_bps_sell: float = 1.0  # 1 bps for index futures; ~25 bps for intraday equity sell
    stamp_bps_buy: float = 0.0  # ~0 for futures; ~1.5 bps for delivery equity

    @classmethod
    def intraday_futures(cls) -> "NSECostModel":
        """Cheapest path — NIFTY futures, day-trade."""
        return cls()

    @classmethod
    def intraday_equity(cls) -> "NSECostModel":
        """Cash equity, square-off same day (MIS). STT 25 bps on sell only."""
        return cls(stt_bps_sell=25.0)

    @classmethod
    def delivery_equity(cls) -> "NSECostModel":
        """Hold-overnight (CNC). STT 10 bps both legs, stamp on buy."""
        return cls(stt_bps_sell=10.0, stamp_bps_buy=1.5)

    @property
    def buy_leg_bps(self) -> float:
        return (
            self.brokerage_bps_per_leg
            + self.exchange_etc_bps_per_leg
            + self.slippage_bps_per_leg
            + self.stamp_bps_buy
        )

    @property
    def sell_leg_bps(self) -> float:
        return (
            self.brokerage_bps_per_leg
            + self.exchange_etc_bps_per_leg
            + self.slippage_bps_per_leg
            + self.stt_bps_sell
        )

    @property
    def round_trip_bps(self) -> float:
        return self.buy_leg_bps + self.sell_leg_bps

    def cost_for_position_change(self, delta_position: float) -> float:
        """Cost (as a fraction of NAV) for changing position by |delta|.

        A flip from -1 to +1 has |delta| = 2: pays a full round-trip (close
        short + open long). A 0 -> +1 entry has |delta| = 1: pays one leg.
        Costs are symmetric (same per leg whether buying or selling) here,
        which is a simplification; STT asymmetry is small relative to slippage.
        """
        # Average per-leg cost simplifies notation; STT asymmetry is small
        # vs slippage so this is fine for index-futures-like execution.
        avg_leg_bps = 0.5 * (self.buy_leg_bps + self.sell_leg_bps)
        return abs(delta_position) * avg_leg_bps / 10_000.0


# ---------------------------------------------------------------------------
# Strategy backtesters
# ---------------------------------------------------------------------------


def backtest_directional(
    forecasts: pd.Series,
    actual_returns: pd.Series,
    cost: NSECostModel,
    initial_position: float = 0.0,
) -> pd.DataFrame:
    """Trade the sign of ``forecasts`` with unit notional.

    At each bar t:
        position_t = sign(forecast_t)        (+1, -1, or 0)
        cost_t     = cost_model.cost_for_position_change(position_t - position_{t-1})
        pnl_t      = position_t * actual_return_t  -  cost_t

    Returns a DataFrame with per-bar position, cost, gross_pnl, net_pnl, equity.
    """
    f = forecasts.to_numpy()
    a = actual_returns.to_numpy()
    n = len(f)
    if n != len(a):
        raise ValueError("forecasts and actual_returns must have equal length.")

    position = np.where(f > 0, 1.0, np.where(f < 0, -1.0, 0.0))
    prev_pos = np.concatenate([[initial_position], position[:-1]])
    delta_pos = position - prev_pos
    costs = np.array([cost.cost_for_position_change(d) for d in delta_pos])

    gross_pnl = position * a
    net_pnl = gross_pnl - costs
    equity = (1.0 + net_pnl).cumprod()

    return pd.DataFrame(
        {
            "forecast": f,
            "actual_ret": a,
            "position": position,
            "delta_pos": delta_pos,
            "cost": costs,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "equity": equity,
        },
        index=forecasts.index,
    )


def backtest_voltarget(
    vol_forecasts: pd.Series,
    actual_returns: pd.Series,
    cost: NSECostModel,
    target_vol_per_bar: float,
    max_leverage: float = 2.0,
    direction: pd.Series | None = None,
) -> pd.DataFrame:
    """Size positions inversely to forecast volatility (vol-targeting).

    position_t = clip( direction_t * (target_vol_per_bar / forecast_sigma_t),
                       -max_leverage, +max_leverage )

    ``direction`` defaults to always +1 (vol-targeted long-only — the classic
    risk-parity style trade). Pass an explicit Series of +1 / -1 / 0 to combine
    a directional model (e.g. ARIMA sign) with vol-based sizing.

    ``target_vol_per_bar`` is the per-bar (not annualized) target. For daily
    bars, 0.005 = 0.5% per day ≈ 8% annualized.
    """
    v = vol_forecasts.to_numpy()
    a = actual_returns.to_numpy()
    n = len(v)
    if n != len(a):
        raise ValueError("vol_forecasts and actual_returns must have equal length.")

    sigma = np.sqrt(np.maximum(v, 1e-20))
    raw_size = target_vol_per_bar / sigma

    if direction is None:
        dir_arr = np.ones(n)
    else:
        if len(direction) != n:
            raise ValueError("direction must align with vol_forecasts.")
        dir_arr = direction.to_numpy()

    position = np.clip(dir_arr * raw_size, -max_leverage, max_leverage)
    prev_pos = np.concatenate([[0.0], position[:-1]])
    delta_pos = position - prev_pos
    costs = np.array([cost.cost_for_position_change(d) for d in delta_pos])

    gross_pnl = position * a
    net_pnl = gross_pnl - costs
    equity = (1.0 + net_pnl).cumprod()

    return pd.DataFrame(
        {
            "forecast_var": v,
            "actual_ret": a,
            "position": position,
            "delta_pos": delta_pos,
            "cost": costs,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "equity": equity,
        },
        index=vol_forecasts.index,
    )


def buy_and_hold(actual_returns: pd.Series) -> pd.DataFrame:
    """Benchmark: always-long, no rebalancing, no costs."""
    a = actual_returns.to_numpy()
    n = len(a)
    return pd.DataFrame(
        {
            "actual_ret": a,
            "position": np.ones(n),
            "delta_pos": np.concatenate([[1.0], np.zeros(n - 1)]),
            "cost": np.zeros(n),
            "gross_pnl": a,
            "net_pnl": a,
            "equity": (1.0 + a).cumprod(),
        },
        index=actual_returns.index,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _annualization_factor(periods_per_year: float) -> float:
    return float(np.sqrt(periods_per_year))


def strategy_metrics(
    bt: pd.DataFrame,
    periods_per_year: float = 252.0,
    risk_free_per_period: float = 0.0,
) -> dict:
    """Compute Sharpe, Sortino, CAGR, max DD, turnover, win-rate.

    Sharpe and Sortino are annualized. CAGR is over the actual elapsed period
    using equity[-1] / equity[0]. Max drawdown is on the equity curve.

    ``periods_per_year``:
        - daily NSE:      252
        - hourly  NSE:    252 * 6.25 ≈ 1575
        - 15-min  NSE:    252 * 25   = 6300
        - 5-min   NSE:    252 * 75   = 18900
    """
    r = bt["net_pnl"].to_numpy()
    eq = bt["equity"].to_numpy()
    n = len(r)

    excess = r - risk_free_per_period
    mu = float(excess.mean())
    sd = float(excess.std(ddof=1)) if n > 1 else 0.0
    ann = _annualization_factor(periods_per_year)
    sharpe = (mu / sd * ann) if sd > 0 else float("nan")

    downside = excess[excess < 0]
    dd_sd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = (mu / dd_sd * ann) if dd_sd > 0 else float("nan")

    # CAGR
    total_return = eq[-1] / eq[0] if eq[0] > 0 else float("nan")
    years = n / periods_per_year
    cagr = total_return ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    # Max drawdown on equity curve
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    max_dd = float(drawdown.min())

    # Turnover (sum of |position changes|) — relates trades to costs
    turnover = float(np.abs(bt["delta_pos"].to_numpy()).sum())

    # Trade-level win rate (only count bars with non-zero position)
    has_pos = bt["position"].to_numpy() != 0
    n_trades = int(has_pos.sum())
    wins = int(((bt["gross_pnl"].to_numpy() > 0) & has_pos).sum())
    win_rate = wins / n_trades if n_trades > 0 else float("nan")

    return {
        "n": n,
        "cagr": cagr,
        "ann_vol": sd * ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "total_return": total_return - 1.0,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "turnover": turnover,
        "total_cost": float(bt["cost"].sum()),
    }
