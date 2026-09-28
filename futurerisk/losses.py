"""Supervision and training objective (S3.7): L_risk, L_evt (with at-risk
masking and class-imbalance handling), L_rank, and the total weighted loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import FutureRiskConfig, DEFAULT_CONFIG


def analytic_risk_loss(
    R: torch.Tensor, y_analytic: torch.Tensor,
    pos_weight: float = 1.0, pos_threshold: float = 0.5,
) -> torch.Tensor:
    """eq:l-risk: L_risk = BCE(R, y_analytic). Both R and y_analytic are
    already probabilities in [0,1] (R from eq:bounded-residual /
    eq:agg, y_analytic from eq:phi).

    S3.7's class-imbalance handling is written only for L_evt, but
    y_analytic is at least as imbalanced in practice: on Bench2Drive-Base
    only ~10.5% of instances have y_analytic > 0.5 (up to ~18% within
    individual scenario types), and plain unweighted BCE on that skew
    systematically pulls R toward the majority-negative side -- observed
    directly as R_predicted's mean sitting around 0.08-0.095 across every
    split despite y_analytic reaching 0.9+ for the riskiest instances. This
    extends the same inverse-frequency upweighting L_evt already gets:
    instances with y_analytic above `pos_threshold` are treated as the rare
    "positive" class and weighted by `pos_weight` (computed by
    `dataset.compute_risk_pos_weight`, analogous to L_evt's pos_weight but
    at the instance level instead of the per-agent event level).
    """
    R = R.clamp(1e-6, 1 - 1e-6)
    weight = torch.where(
        y_analytic > pos_threshold,
        torch.full_like(y_analytic, pos_weight),
        torch.ones_like(y_analytic),
    )
    return F.binary_cross_entropy(R, y_analytic, weight=weight)


def event_time_bin_index(T_evt: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Map a (possibly +inf) event time to a bin index in [0, K], where
    index k in [0, K-1] means the event fell in (b_k, b_{k+1}] and index K
    means censored (no event within [0, H], T_evt = +inf or > b_K)."""
    K = bin_edges.numel() - 1
    b_k = bin_edges[1:]                      # (K,), b_1..b_K
    # searchsorted: first index where T_evt <= b_k  (right-closed bins)
    idx = torch.searchsorted(b_k.contiguous(), T_evt.contiguous().clamp_max(b_k[-1] + 1e9))
    idx = torch.where(torch.isinf(T_evt) | (T_evt > b_k[-1]), torch.full_like(idx, K), idx)
    return idx


def event_hazard_loss(
    q: torch.Tensor,                      # (B, K) predicted hazards
    T_evt: torch.Tensor,                  # (B,) event time per agent, +inf if none
    bin_edges: torch.Tensor,              # (K+1,)
    pos_weight: float = 1.0,
    agent_mask: Optional[torch.Tensor] = None,  # (B,) bool, True = real agent (not padding)
) -> torch.Tensor:
    """eq:l-evt, discrete-time hazard / survival loss with at-risk masking:

    L_evt = -(1/J) * sum_j sum_{k=1}^K [
        1[T_evt^j in (b_{k-1}, b_k]] * log q_k^j
        + 1[T_evt^j > b_k] * log(1 - q_k^j)
    ]

    "At-risk masking": bins strictly after the event (or, for a censored
    agent, none are excluded) stop contributing. Class-imbalance handling
    (S3.7): positive-class upweighting via `pos_weight` (e.g.
    inverse-frequency, computed per dataset by the caller and passed in).
    """
    B, K = q.shape
    event_bin = event_time_bin_index(T_evt, bin_edges)          # (B,), in [0, K]
    event_occurred = event_bin < K                               # (B,) bool

    k_idx = torch.arange(K, device=q.device).unsqueeze(0).expand(B, K)   # (B, K), 0-indexed bin positions
    at_risk_mask = (k_idx <= event_bin.unsqueeze(1)).float()             # (B, K)
    target = (event_occurred.unsqueeze(1) & (k_idx == event_bin.unsqueeze(1))).float()  # (B, K)

    q_clamped = q.clamp(1e-6, 1 - 1e-6)
    pos_term = pos_weight * target * torch.log(q_clamped)
    neg_term = (1 - target) * torch.log(1 - q_clamped)
    per_bin_loss = -(pos_term + neg_term) * at_risk_mask          # (B, K)

    per_agent_loss = per_bin_loss.sum(dim=1)                       # (B,)
    if agent_mask is not None:
        per_agent_loss = per_agent_loss * agent_mask.float()
        denom = agent_mask.float().sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(B), device=q.device)
    return per_agent_loss.sum() / denom


def safe_ttc_saturation_penalty(
    g_mu_raw: torch.Tensor, g_sigma_raw: torch.Tensor,
    agent_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Not part of instruction.txt's spec -- added after
    `tau_safe_vs_tau_phy.py` (S-five.2) found `mu_s`/`sigma_s` collapsed to
    within 0.1% of their bound for ~98% of instances regardless of context,
    across every interaction type and relative-speed bin.

    Root cause (see `carm_model.InteractionEncoder`'s `out_norm` docstring
    for the full mechanism): `g_mu`/`g_sigma` get NO direct supervision
    (S3.6.3: only through the composition), and are trained jointly in
    Stage 1 alongside the much higher-capacity 25-dim `hazard_head`, which
    can absorb essentially all of L_risk's discriminative signal on its
    own -- leaving gradient descent with no real incentive to keep these
    two 1-scalar-output heads within their nonlinearities' well-conditioned
    range (tanh/sigmoid are both close to flat once |raw| exceeds ~2-3).
    A LayerNorm on the shared representation `h` (added alongside this)
    reduced but did not fix the collapse on its own (h's own scale doesn't
    determine what scale g_mu/g_sigma's own weights settle into).

    This is a plain L2-to-zero penalty on the raw pre-nonlinearity outputs
    -- cheap, and doesn't touch what mu_s/sigma_s are *supervised toward*
    (there still isn't a direct target for them), just discourages the
    network from finding a solution that only works by pushing them into
    a saturated, context-blind corner. `agent_mask` excludes padded agent
    slots the same way `event_hazard_loss` does."""
    sq = g_mu_raw ** 2 + g_sigma_raw ** 2
    if agent_mask is not None:
        mask = agent_mask.float()
        denom = mask.sum().clamp_min(1.0)
        return (sq * mask).sum() / denom
    return sq.mean()


def rank_loss(R_worse: torch.Tensor, R_better: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """eq:l-rank, margin ranking loss for matched-scenario multi-policy
    pairs (Bench2Drive only): L_rank = max(0, gamma - (R_worse - R_better))
    when `worse` is the known-worse-safety-outcome instance."""
    return torch.clamp(gamma - (R_worse - R_better), min=0.0).mean()


def total_loss(
    L_evt: Optional[torch.Tensor] = None,
    L_risk: Optional[torch.Tensor] = None,
    L_rank: Optional[torch.Tensor] = None,
    L_saturation: Optional[torch.Tensor] = None,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """eq:total-loss: L = lambda_evt*L_evt + lambda_risk*L_risk +
    lambda_rank*L_rank. Any term may be omitted (e.g. Stage 1 excludes
    L_rank; Stage 2 excludes L_evt -- S3.8). `L_saturation` is the
    post-hoc addition from `safe_ttc_saturation_penalty` (not in the
    original spec, Stage 1 only -- Stage 2 doesn't touch g_mu/g_sigma)."""
    total = torch.tensor(0.0)
    device = None
    for term, weight in ((L_evt, cfg.lambda_evt), (L_risk, cfg.lambda_risk), (L_rank, cfg.lambda_rank),
                         (L_saturation, cfg.lambda_saturation)):
        if term is not None:
            device = term.device
            total = total.to(device) + weight * term
    return total
