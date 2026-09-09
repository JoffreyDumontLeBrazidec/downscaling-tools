"""The station head and its loss. UNTRAINED: this module has never been fitted.

Nothing in this file has seen data. It exists so that the smoke tests of section 8
of the design note can be run before any training, and so that the next session
starts from a model whose shapes and loss are already checked rather than from a
blank page. No checkpoint of this model exists.

The model follows section 6 of the design note. For one station and one member it
receives the predicted values at the twelve nearest output points, the interpolated
AIFS input at the same points, the static features of the station relative to the
output grid, and the time of day, and returns one number, the value that station is
predicted to measure. Members are processed independently and share every weight,
so an ensemble of any size can be passed through the same network; the ensemble
only enters through the loss.

The loss is the fair kernel form of the continuous ranked probability score. For an
ensemble of m members x_1 .. x_m and one observation y it is

    CRPS = (1/m) sum_i |x_i - y|  -  (1 / (2 m (m - 1))) sum_i sum_j |x_i - x_j|

The second term carries the factor 1/(m(m-1)) rather than 1/m^2, which is what
makes the estimator fair, meaning unbiased for the score of the underlying
distribution rather than optimistic for a small ensemble. With a single member the
spread term has no pairs to average and the score is exactly the absolute error,
which is one of the smoke tests of the design note.
"""
from __future__ import annotations

import torch
from torch import nn


def fair_crps(members: torch.Tensor, observation: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Fair kernel CRPS, averaged over the entries the mask keeps.

    members: (n, m) predictions, one row per case-station and one column per member
    observation: (n,) the single verifying value
    mask: (n,) boolean, False where the observation is missing
    """
    if members.dim() != 2:
        raise ValueError("members must be (n, m)")
    n, m = members.shape
    if observation.shape != (n,):
        raise ValueError("observation must be (n,)")
    skill = (members - observation.unsqueeze(1)).abs().mean(dim=1)
    if m > 1:
        pair = (members.unsqueeze(2) - members.unsqueeze(1)).abs().sum(dim=(1, 2))
        spread = pair / (2.0 * m * (m - 1))
    else:
        spread = torch.zeros_like(skill)
    per_row = skill - spread
    if mask is None:
        return per_row.mean()
    mask = mask.to(per_row.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (per_row * mask).sum() / denom


class StationHead(nn.Module):
    """A multilayer perceptron over one station's neighbourhood. UNTRAINED."""

    def __init__(self, n_neighbourhood: int, n_static: int, hidden: int = 256,
                 depth: int = 3, n_time: int = 2) -> None:
        super().__init__()
        n_in = n_neighbourhood + n_static + n_time
        layers: list[nn.Module] = []
        width = n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.n_neighbourhood = n_neighbourhood
        self.n_static = n_static
        self.n_time = n_time

    def forward(self, neighbourhood: torch.Tensor, static: torch.Tensor,
                time_of_day: torch.Tensor) -> torch.Tensor:
        """neighbourhood: (n, m, n_neighbourhood); static: (n, n_static);
        time_of_day: (n, n_time). Returns (n, m)."""
        n, m, _ = neighbourhood.shape
        static_e = static.unsqueeze(1).expand(n, m, static.shape[-1])
        time_e = time_of_day.unsqueeze(1).expand(n, m, time_of_day.shape[-1])
        x = torch.cat([neighbourhood, static_e, time_e], dim=-1)
        return self.net(x.reshape(n * m, -1)).reshape(n, m)
