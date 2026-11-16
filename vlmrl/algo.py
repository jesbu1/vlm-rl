"""GRPO-family pieces: group advantages, DAPO group filter, policy-gradient loss.

PEAR recipe (from the blog): GRPO + DAPO group filter + Score Centering; REINFORCE policy loss
("bypass mode", no PPO clipping); token-mean aggregation; no KL, no entropy bonus; one optimizer
update per step.
"""
from __future__ import annotations

import numpy as np
import torch


def group_advantages(scores: list[list[float]], mode: str = "center", eps: float = 1e-6) -> list[list[float]]:
    """Per-group baselines.

    * ``center`` : A = s - mean(s)              (score centering; Dr.GRPO-style, no std division)
    * ``zscore`` : A = (s - mean) / (std + eps) (original GRPO)
    * ``rloo``   : A = s - mean of the *other* G-1 members (unbiased leave-one-out baseline)

    NOTE: the PEAR blog cites "Score Centering" (Marek & Ryabinin) without spelling it out; ``center``
    is the plain mean-centering reading of that name. Swap in the exact form here if needed.
    """
    out = []
    for g in scores:
        s = np.asarray(g, dtype=np.float64)
        if mode == "center":
            a = s - s.mean()
        elif mode == "zscore":
            a = (s - s.mean()) / (s.std() + eps)
        elif mode == "rloo":
            n = len(s)
            a = s - (s.sum() - s) / max(n - 1, 1)
        else:
            raise ValueError(mode)
        out.append(a.tolist())
    return out


def keep_group(scores: list[float], eps: float = 1e-6) -> bool:
    """DAPO dynamic-sampling filter: drop groups with no learning signal (all scores equal)."""
    return float(np.std(scores)) > eps


def policy_loss_sum(logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                    old_logp: torch.Tensor | None = None, clip: tuple[float, float] | None = None
                    ) -> tuple[torch.Tensor, dict]:
    """Summed (not averaged) per-token loss; the caller divides by the *global* token count.

    Summing per micro-batch and normalising once by the total number of response tokens in the
    whole step (FeynRL-style) gives the exact token-mean gradient regardless of how sequences were
    split into micro-batches. A per-micro-batch mean would overweight short micro-batches.

    * ``old_logp is None`` -> REINFORCE (PEAR "bypass"): L = -A * log pi. On-policy with one update
      per step this has the same gradient as the PPO surrogate at ratio=1.
    * otherwise PPO/DAPO clipped surrogate with asymmetric (clip-higher) bounds.
    """
    mask = mask.to(logp.dtype)
    if old_logp is None:
        per_tok = -adv * logp
        stats = {}
    else:
        ratio = torch.exp(logp - old_logp)
        lo, hi = clip if clip is not None else (0.2, 0.28)
        per_tok = -torch.minimum(ratio * adv, ratio.clamp(1 - lo, 1 + hi) * adv)
        with torch.no_grad():
            clipped = ((ratio < 1 - lo) | (ratio > 1 + hi)).to(logp.dtype)
            stats = {"clipfrac_sum": (clipped * mask).sum().item()}
    return (per_tok * mask).sum(), stats
