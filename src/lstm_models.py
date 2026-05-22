"""PyTorch LSTM models for intraday return prediction.

Two seq-to-one models that share an identical recurrent trunk and differ only
in their output head:

    LSTMRegressor  -> one scalar (the percent return of the next bar)
    LSTMClassifier -> one logit  (up/down direction of the next bar)

Design choices, all deliberate for this small-data regime (~1200 training bars
per stock):

* Seq-to-one: the LSTM consumes all ``seq_len`` bars but only the hidden state
  at the FINAL timestep feeds the output head. Standard for "given a window,
  predict the next step" and avoids inventing a target for every intermediate
  bar.
* Small by default: 1 layer, 32 hidden units, dropout 0.2. With ~1200 samples a
  larger network memorizes noise — it will drive training loss down while the
  held-out paper-trade window stays at coin-flip. Ablate UP only after the
  small model shows signal.
* Classifier emits a raw logit, not a probability. Pair it with
  BCEWithLogitsLoss (numerically stabler than sigmoid + BCE). Apply sigmoid
  only at inference when you actually want P(up).
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM -> last hidden state -> Linear -> scalar return prediction.

    Forward input : (batch, seq_len, n_features)
    Forward output: (batch,)  -- one predicted value per window, in the same
                    unit as the training target (percent return, per Option B).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch only applies LSTM's internal dropout BETWEEN stacked layers,
        # so a 1-layer LSTM ignores the dropout arg (it would warn). Guard it.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # A dropout layer on the final hidden state gives us regularization even
        # in the 1-layer case (where LSTM internal dropout is disabled).
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output: (batch, seq_len, hidden); we want the last timestep only.
        output, _ = self.lstm(x)
        last = output[:, -1, :]              # (batch, hidden)
        last = self.head_dropout(last)
        pred = self.head(last)               # (batch, 1)
        return pred.squeeze(-1)              # (batch,)


class LSTMClassifier(nn.Module):
    """LSTM -> last hidden state -> Linear -> single logit (up/down).

    Forward input : (batch, seq_len, n_features)
    Forward output: (batch,)  -- raw logits. Use BCEWithLogitsLoss for training;
                    apply torch.sigmoid() at inference to get P(next bar up).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        last = self.head_dropout(last)
        logit = self.head(last)              # (batch, 1)
        return logit.squeeze(-1)             # (batch,)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count — a sanity check against overfitting.

    Print this next to your training-set size: if params >> samples, expect the
    model to memorize. For 1 layer / 32 units / ~6 features this is ~5k params.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
