"""Conditional value-at-risk (CVaR), used for safety responsibility (Eq. 3).

We use the standard Rockafellar & Uryasev convention where ``alpha`` is a
confidence level: CVaR_alpha(C) is the expected value of C in its upper
(1 - alpha) tail. This is the convention consistent with the paper's own
stated limit, "in the extreme case when alpha -> 1, the safety
responsibility is the maximum safety outcome decrease" (Sec III-B): as
alpha -> 1 the tail shrinks to a single point, the maximum. The in-text
formula in the paper (integrating VaR over gamma in [1-alpha, 1] and
dividing by alpha) is the mirror-image convention for a *lower*-tail loss
and does not itself reduce to that stated limit; we implement the
upper-tail convention so the code matches the paper's documented behavior
and its experiment (``alpha=0.1`` for safety responsibility, i.e. CVaR
close to the mean, only lightly emphasizing the tail).
"""

import torch
from torch import Tensor


def value_at_risk(samples: Tensor, alpha: float) -> Tensor:
    """VaR_alpha(C): the alpha-quantile of ``samples`` (last dim)."""
    return torch.quantile(samples, alpha, dim=-1)


def cvar(samples: Tensor, alpha: float) -> Tensor:
    """CVaR_alpha(C) := E[C | C >= VaR_alpha(C)].

    Args:
        samples: [..., N] a batch of scalar samples (e.g. safety-margin
            decreases from N counterfactual motions).
        alpha: confidence level in (0, 1). alpha -> 1 emphasizes only the
            single worst sample (the max); alpha -> 0 averages over the
            whole set (the mean).

    Returns:
        [...] the CVaR of each batch element.
    """
    if samples.shape[-1] == 0:
        return torch.zeros(samples.shape[:-1], device=samples.device)
    if samples.shape[-1] == 1:
        return samples.squeeze(-1)

    var = value_at_risk(samples, alpha).unsqueeze(-1)  # [..., 1]
    tail_mask = samples >= var
    # guard against an empty tail from ties/degenerate quantiles
    has_tail = tail_mask.any(dim=-1, keepdim=True)
    tail_mask = tail_mask | (~has_tail & (samples == samples.max(dim=-1, keepdim=True).values))

    count = tail_mask.sum(dim=-1).clamp(min=1)
    total = (samples * tail_mask).sum(dim=-1)
    return total / count
