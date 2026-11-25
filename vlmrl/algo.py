"""GRPO-family pieces: group advantages, DAPO group filter, policy-gradient loss.

PEAR recipe (from the blog): GRPO + DAPO group filter + Score Centering; REINFORCE policy loss
("bypass mode", no PPO clipping); token-mean aggregation; no KL, no entropy bonus; one optimizer
update per step.
"""
from __future__ import annotations

import numpy as np
import torch


def group_advantages(scores: list[list[float]], mode: str = "zscore", eps: float = 1e-6) -> list[list[float]]:
    """Per-group baselines.

    * ``zscore`` : A = (s - mean) / (std + eps)   GRPO; PEAR Eq. (1) -- the default
    * ``center`` : A = s - mean(s)                 Dr.GRPO-style (no std division)
    * ``rloo``   : A = s - mean of the *other* G-1 members (leave-one-out baseline)
    * ``reinforce_pp`` : A = s - mean(s) here; the trainer then divides every advantage of the step by their std over the
      whole batch (``batch_normalize``): REINFORCE++-baseline (Hu et al., 2025), which avoids GRPO's per-group std, a
      division that blows a group of near-equal rollouts up to full-size advantages

    (PEAR's "score centering" is *not* an advantage baseline; it is a gradient correction for
    rollout/learner policy mismatch -- see ``score_centering_term``.)
    """
    out = []
    for g in scores:
        s = np.asarray(g, dtype=np.float64)
        if mode in ("center", "reinforce_pp"):
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


def batch_normalize(advantages: list[float], eps: float = 1e-6) -> list[float]:
    """Global (whole-batch) advantage normalization, REINFORCE++ style: (A - mean) / std over every rollout of the step."""
    a = np.asarray(advantages, dtype=np.float64)
    return ((a - a.mean()) / (a.std() + eps)).tolist() if len(a) else []


def keep_group(scores: list[float], eps: float = 1e-6) -> bool:
    """DAPO dynamic-sampling filter: drop groups with no learning signal (all scores equal)."""
    return float(np.std(scores)) > eps


def score_centering_term(logp_at_ids: torch.Tensor, mu_probs: torch.Tensor) -> torch.Tensor:
    """Score centering (Marek & Ryabinin; PEAR Eq. 2): a per-token surrogate whose gradient is
    E_{v~mu}[grad log pi_theta(v|c)], the expected score under the *rollout* policy mu.

    PEAR's estimator is  g = 1/N sum_i A_i sum_t [grad log pi(y_t|c_t) - E_{v~mu(.|c_t)} grad log pi(v|c_t)].
    Since sum_v pi(v) grad log pi(v) = 0, the expectation equals sum_v (mu_v - pi_v) grad log pi_v; we
    evaluate it on mu's top-k tokens, so the term is *exactly zero when mu == pi* (on-policy) for any k,
    and the only approximation is the tail mass outside the top-k.

    logp_at_ids [B,T,k]: log pi_theta at mu's top-k ids (with grad); mu_probs [B,T,k]: their mu probs.
    """
    pi = logp_at_ids.detach().exp()
    return ((mu_probs - pi) * logp_at_ids).sum(-1)


def policy_loss_sum(logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                    old_logp: torch.Tensor | None = None, clip: tuple[float, float] | None = None,
                    centering: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Summed (not averaged) per-token loss; the caller divides by the *global* token count.

    Summing per micro-batch and normalising once by the total number of response tokens in the
    whole step (FeynRL-style) gives the exact token-mean gradient regardless of how sequences were
    split into micro-batches. A per-micro-batch mean would overweight short micro-batches.

    * ``old_logp is None`` -> REINFORCE (PEAR "bypass"): L = -A * (log pi - centering), where
      ``centering`` is the optional score-centering surrogate (``score_centering_term``). On-policy
      with one update per step this has the same gradient as the PPO surrogate at ratio=1.
    * otherwise PPO/DAPO clipped surrogate with asymmetric (clip-higher) bounds.
    """
    mask = mask.to(logp.dtype)
    if old_logp is None:
        per_tok = -adv * (logp - centering) if centering is not None else -adv * logp
        stats = {}
    else:
        ratio = torch.exp(logp - old_logp)
        lo, hi = clip if clip is not None else (0.2, 0.28)
        per_tok = -torch.minimum(ratio * adv, ratio.clamp(1 - lo, 1 + hi) * adv)
        with torch.no_grad():
            clipped = ((ratio < 1 - lo) | (ratio > 1 + hi)).to(logp.dtype)
            stats = {"clipfrac_sum": (clipped * mask).sum().item()}
    return (per_tok * mask).sum(), stats


def kl_k3_sum(logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over masked tokens of the k3 estimator of KL(pi || pi_ref) on tokens sampled from pi (Schulman's
    approximation, as in GRPO): exp(r) - r - 1 with r = ref_logp - logp. Non-negative, zero iff the two agree on the
    sampled token, and its gradient w.r.t. logp is 1 - exp(r), which pulls logp back toward ref_logp."""
    r = (ref_logp.detach() - logp).clamp(-20, 20)
    return ((torch.exp(r) - r - 1) * mask).sum()
