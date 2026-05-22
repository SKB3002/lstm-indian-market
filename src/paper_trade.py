"""Bar-by-bar paper-trade simulator with explicit per-trade logging.

Difference from backtest.py:
- backtest.py is vectorized — fast, but produces only an aggregate frame.
- paper_trade.py is *stepwise* — slower, but you get to see (and log) every
  decision exactly as it would happen at the moment a new bar arrives.

This is the right shape for two things:
1. Replaying a held-out window in "paper trading" mode and producing a trade
   blotter the user can read line-by-line.
2. Plugging into a live loop that polls yfinance every 15 minutes during
   market hours (a thin wrapper on top of this loop is all that's needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.backtest import NSECostModel


@dataclass
class Bar:
    """One OHLCV bar, exchange-timestamped."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeRecord:
    """One line in the paper-trade blotter."""

    ts: pd.Timestamp
    close: float
    realized_logret: float          # actual return realized on this bar
    forecast_ret: float | None      # ARIMA forecast (the signal at bar start, for THIS bar)
    forecast_var: float | None      # GARCH forecast variance (for THIS bar)
    target_position: float          # what the rule wants us to be in for the NEXT bar
    prev_position: float
    delta_pos: float
    cost: float
    gross_pnl: float
    net_pnl: float
    equity: float
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "close": self.close,
            "realized_logret": self.realized_logret,
            "forecast_ret": self.forecast_ret,
            "forecast_var": self.forecast_var,
            "target_position": self.target_position,
            "prev_position": self.prev_position,
            "delta_pos": self.delta_pos,
            "cost": self.cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "equity": self.equity,
            "note": self.note,
        }


# A SignalRule maps (forecast_ret, forecast_var) -> desired position.
# Either argument may be None if the corresponding model isn't being used.
SignalRule = Callable[[float | None, float | None], float]


def rule_arima_directional(forecast_ret: float | None, _: float | None) -> float:
    """+1 if forecast > 0, -1 if < 0, 0 if exactly 0."""
    if forecast_ret is None or forecast_ret == 0:
        return 0.0
    return 1.0 if forecast_ret > 0 else -1.0


def make_voltarget_long_rule(target_vol_per_bar: float, max_leverage: float = 2.0) -> SignalRule:
    """Always-long sizing inversely proportional to forecast volatility."""
    def rule(_: float | None, forecast_var: float | None) -> float:
        if forecast_var is None or forecast_var <= 0:
            return 0.0
        sigma = float(np.sqrt(forecast_var))
        return float(np.clip(target_vol_per_bar / sigma, -max_leverage, max_leverage))
    return rule


def make_combined_rule(target_vol_per_bar: float, max_leverage: float = 2.0) -> SignalRule:
    """ARIMA sign x GARCH-scaled magnitude."""
    def rule(forecast_ret: float | None, forecast_var: float | None) -> float:
        if forecast_ret is None or forecast_var is None or forecast_var <= 0:
            return 0.0
        direction = 1.0 if forecast_ret > 0 else (-1.0 if forecast_ret < 0 else 0.0)
        sigma = float(np.sqrt(forecast_var))
        return float(np.clip(direction * target_vol_per_bar / sigma, -max_leverage, max_leverage))
    return rule


@dataclass
class PaperTrader:
    """Bar-by-bar paper-trading simulator.

    Lifecycle:
        - construct with a SignalRule and a cost model
        - feed the previous-bar forecast(s) and the current bar's realised
          return into ``observe_bar()``; it records the trade and returns the
          new target position (i.e. what position the trader holds *going
          into* the next bar)
    """

    rule: SignalRule
    cost: NSECostModel
    name: str = "strategy"
    position: float = 0.0
    equity: float = 1.0
    blotter: list[TradeRecord] = field(default_factory=list)

    def observe_bar(
        self,
        ts: pd.Timestamp,
        close: float,
        realized_logret: float,
        forecast_ret_for_this_bar: float | None = None,
        forecast_var_for_this_bar: float | None = None,
        forecast_ret_for_next_bar: float | None = None,
        forecast_var_for_next_bar: float | None = None,
        volume: float = 0.0,
    ) -> float:
        """Process the arrival of one bar.

        The chronology of a 15-min bar at, say, 11:30 IST:
            - At 11:30:00 the *previous* bar (11:15) closes.
            - Our position over the [11:15, 11:30) interval was whatever we
              decided at the *11:15* close, based on the forecast we made
              *at* the 11:15 close for the [11:15, 11:30) bar.
            - The realized return for [11:15, 11:30) is log(close_11:30 /
              close_11:15).
            - We now multiply our held position by that return = gross_pnl.
            - Then we generate a NEW forecast (for the [11:30, 11:45) bar)
              and update our position accordingly, paying transaction cost
              on the position change.

        Args:
            ts: timestamp of the bar that just closed.
            close: close price of this bar.
            realized_logret: log-return realized on this bar (vs prior close).
            forecast_ret_for_this_bar: the ARIMA forecast we *had* for this
                bar at the prior bar's close. Used only for the blotter.
            forecast_var_for_this_bar: same, for GARCH variance.
            forecast_ret_for_next_bar: the NEW forecast we now have for the
                next bar. Drives the new position.
            forecast_var_for_next_bar: same, GARCH.

        Returns:
            The new target position to hold into the next bar.
        """
        # 1. Realize P&L on the position we held into this bar.
        gross = self.position * realized_logret

        # 2. Decide new position from the just-generated forecasts.
        new_position = self.rule(forecast_ret_for_next_bar, forecast_var_for_next_bar)
        delta_pos = new_position - self.position
        cost = self.cost.cost_for_position_change(delta_pos)
        net = gross - cost
        self.equity *= 1.0 + net

        note_parts = []
        if delta_pos != 0:
            note_parts.append(f"rebalance Δ={delta_pos:+.3f}")
        if forecast_ret_for_next_bar is not None and forecast_ret_for_next_bar != 0:
            note_parts.append("long" if forecast_ret_for_next_bar > 0 else "short")
        note = " | ".join(note_parts) if note_parts else "hold"

        self.blotter.append(
            TradeRecord(
                ts=ts,
                close=close,
                realized_logret=realized_logret,
                forecast_ret=forecast_ret_for_this_bar,
                forecast_var=forecast_var_for_this_bar,
                target_position=new_position,
                prev_position=self.position,
                delta_pos=delta_pos,
                cost=cost,
                gross_pnl=gross,
                net_pnl=net,
                equity=self.equity,
                note=note,
            )
        )
        self.position = new_position
        return new_position

    def blotter_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.as_dict() for t in self.blotter]).set_index("ts")

    def summary(self, periods_per_year: float) -> dict:
        df = self.blotter_df()
        r = df["net_pnl"].to_numpy()
        eq = df["equity"].to_numpy()
        n = len(r)

        mu, sd = float(r.mean()), float(r.std(ddof=1)) if n > 1 else 0.0
        ann = float(np.sqrt(periods_per_year))
        sharpe = (mu / sd * ann) if sd > 0 else float("nan")

        downside = r[r < 0]
        dd_sd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
        sortino = (mu / dd_sd * ann) if dd_sd > 0 else float("nan")

        total_return = eq[-1] - 1.0
        running_max = np.maximum.accumulate(eq)
        max_dd = float((eq / running_max - 1.0).min())

        trades_with_position = (df["target_position"] != 0).sum()
        wins = int(((df["gross_pnl"] > 0) & (df["prev_position"] != 0)).sum())
        n_realized = int((df["prev_position"] != 0).sum())
        win_rate = wins / n_realized if n_realized > 0 else float("nan")
        n_flips = int((df["delta_pos"] != 0).sum())

        return {
            "name": self.name,
            "n_bars": n,
            "n_trades_with_pos": int(trades_with_position),
            "n_position_changes": n_flips,
            "win_rate": win_rate,
            "total_return": total_return,
            "ann_vol": sd * ann,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "total_cost": float(df["cost"].sum()),
            "final_equity": float(eq[-1]),
        }
