"""Context-Adaptive Risk Model (CARM), S3.6.

Requires torch. All equation references are to the numbered equations in
S3.6 (eq:encoder, eq:hazard, eq:safettc, eq:composition, eq:bounded-residual,
eq:agg).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FutureRiskConfig, DEFAULT_CONFIG


def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 1, dropout: float = 0.0) -> nn.Sequential:
    """dropout is applied after each hidden ReLU (never after the final
    output Linear, where it would directly corrupt the head's output
    rather than regularize a learned representation)."""
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    """Simple additive-attention pooling over a sequence, used as an
    alternative to final-hidden-state concatenation (eq:encoder note:
    "pooled sequence representation (e.g. final hidden state concatenation
    of both directions, or attention pooling)")."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # seq: (B, K, dim); mask: (B, K) bool, True = valid
        logits = self.score(seq).squeeze(-1)  # (B, K)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bk,bkd->bd", weights, seq)


class InteractionEncoder(nn.Module):
    """f_enc, eq:encoder.

    h^j = f_enc(x_kin^j, X_int^j, x_ctx^j)

    implemented as a bidirectional GRU over X_int^j, with x_kin^j, x_ctx^j
    encoded by small MLPs and fused (concat + MLP) with the pooled sequence
    representation. Note: dataset/domain identity is explicitly excluded
    from all three inputs (S3.6.1).
    """

    def __init__(
        self,
        d_kin: int,
        d_int: int,
        d_ctx: int,
        hidden_dim: int = 64,
        rnn_hidden: int = 64,
        pooling: str = "attention",  # "attention" or "final_state"
        rnn_type: str = "gru",       # "gru" or "lstm"
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pooling_mode = pooling
        self.kin_mlp = _mlp(d_kin, hidden_dim, hidden_dim, dropout=dropout)
        self.ctx_mlp = _mlp(d_ctx, hidden_dim, hidden_dim, dropout=dropout)

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_int, hidden_size=rnn_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        seq_repr_dim = 2 * rnn_hidden
        if pooling == "attention":
            self.pool = AttentionPool(seq_repr_dim)

        self.fusion = _mlp(hidden_dim + hidden_dim + seq_repr_dim, hidden_dim, hidden_dim, dropout=dropout)
        # Normalize the shared representation before any head reads it.
        #
        # Bug found via tau_safe_vs_tau_phy.py (S-five.2 experiment): without
        # this, h's norm grows unconstrained during training (observed mean
        # ~12.3 for a 64-dim vector on the un-normalized checkpoint) and
        # SafeTTCHead's g_mu/g_sigma -- small, 1-scalar-output heads trained
        # jointly in Stage 1 alongside the much higher-capacity 25-dim
        # hazard_head, with NO direct supervision of their own (S3.6.3: only
        # through the composition) -- had every incentive to drift and none
        # to stay well-scaled, since gradient descent could push essentially
        # all of L_risk's discriminative signal through hazard_head instead.
        # Result: g_mu(h) raw outputs ranged -12.2 to -1.6 (98.5% with
        # |raw|>3, deep in tanh's saturated regime), collapsing
        # delta_max*tanh(g_mu(h)) to within 0.001 of its floor for nearly
        # every instance regardless of context -- the "context-adaptive"
        # correction the whole SafeTTCHead exists for was being computed but
        # then thrown away by the saturating nonlinearity. The bounded
        # `residual_head` (trained ALONE in Stage 2, encoder/hazard/safe_ttc
        # frozen, so no such escape route existed) did NOT show this
        # collapse -- confirming the mechanism, not just the symptom.
        # LayerNorm keeps every downstream head's input on a fixed, bounded
        # scale regardless of what the shared encoder settles into; its own
        # learnable affine params still let hazard_head (or anything else)
        # recover whatever scale it actually needs.
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(
        self,
        x_kin: torch.Tensor,             # (B, d_kin)
        X_int: torch.Tensor,             # (B, K, d_int)
        x_ctx: torch.Tensor,             # (B, d_ctx)
        seq_mask: Optional[torch.Tensor] = None,  # (B, K) bool, True=valid bin
    ) -> torch.Tensor:
        h_kin = self.kin_mlp(x_kin)
        h_ctx = self.ctx_mlp(x_ctx)

        seq_out, final_state = self.rnn(X_int)  # seq_out: (B, K, 2*rnn_hidden)
        if self.pooling_mode == "attention":
            h_seq = self.pool(seq_out, mask=seq_mask)
        else:
            if isinstance(final_state, tuple):  # LSTM -> (h_n, c_n)
                final_state = final_state[0]
            # final_state: (num_layers*2, B, rnn_hidden) -> last layer, both dirs
            h_fwd, h_bwd = final_state[-2], final_state[-1]
            h_seq = torch.cat([h_fwd, h_bwd], dim=-1)

        fused = torch.cat([h_kin, h_ctx, h_seq], dim=-1)
        return self.out_norm(self.fusion(fused))  # h^j, (B, hidden_dim)


class EventTimeHazardHead(nn.Module):
    """g_evt, eq:hazard: discrete-time survival head with K sigmoid
    outputs (DeepSurv-style discrete hazard parameterization)."""

    def __init__(self, h_dim: int, num_bins: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.mlp = _mlp(h_dim, hidden, num_bins, dropout=dropout)

    def forward(self, h: torch.Tensor):
        """Returns q_k, p_k, F_k, each (B, K)."""
        logits = self.mlp(h)                       # (B, K)
        q = torch.sigmoid(logits)
        # survival up to (excluding) bin k: prod_{l<k} (1 - q_l), exclusive cumprod
        one_minus_q = 1 - q
        surv_excl = torch.cumprod(
            torch.cat([torch.ones_like(one_minus_q[:, :1]), one_minus_q[:, :-1]], dim=1),
            dim=1,
        )
        p = q * surv_excl                            # (B, K)
        F = torch.cumsum(p, dim=1)                   # (B, K)
        return q, p, F


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SafeTTCHead(nn.Module):
    """(g_mu, g_sigma), eq:safettc: context-adaptive safe-TTC head.

    mu_s^j = log(tau_phy^j) + Delta_max * tanh(g_mu(h^j))
    sigma_s^j = sigma_min + (sigma_max - sigma_min) * sigma(g_sigma(h^j))
    T_s^j ~ LogNormal(mu_s^j, sigma_s^j^2)

    tau_phy^j is treated as a per-bin sequence tau_phy^j(b_k) (S3.6.3
    "critical implementation detail" note), so mu_s is (B, K); g_mu/g_sigma
    themselves are scalar-per-(agent,instance) corrections (not bin
    indexed), applied uniformly on top of the bin-varying physical prior.

    `mu_enabled`/`sigma_enabled` (default True) ablate the two learned
    quantities in eq:safettc independently, for the full-R decomposition
    ablation (root.tex Table~\\ref{tab:residual_ablation}'s extended rows):
    with `mu_enabled=False`, Delta_mu's correction is forced to 0 (mu_s
    collapses to the raw physical prior log(tau_phy) exactly, g_mu is never
    even instantiated so it cannot be trained); with `sigma_enabled=False`,
    sigma_s is forced to the fixed midpoint (sigma_min+sigma_max)/2 -- a
    reasonable constant with no learning and no physics anchor at all
    (c_sigma carries no anchor by construction either way, so there is no
    "raw" fallback for it the way there is for mu_s) -- g_sigma likewise
    never instantiated. Both default True, matching the full model.
    """

    def __init__(self, h_dim: int, delta_max: float, sigma_min: float, sigma_max: float, hidden: int = 32,
                 dropout: float = 0.0, mu_enabled: bool = True, sigma_enabled: bool = True):
        super().__init__()
        self.mu_enabled = mu_enabled
        self.sigma_enabled = sigma_enabled
        if mu_enabled:
            self.g_mu = _mlp(h_dim, hidden, 1, dropout=dropout)
        if sigma_enabled:
            self.g_sigma = _mlp(h_dim, hidden, 1, dropout=dropout)
        self.delta_max = delta_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._sigma_fixed = (sigma_min + sigma_max) / 2.0

    def forward(self, h: torch.Tensor, tau_phy_seq: torch.Tensor):
        """
        h: (B, h_dim)
        tau_phy_seq: (B, K) physical-prior safe-TTC evaluated at each bin
            midpoint/edge b_k (must be > 0).
        Returns mu_s (B, K), sigma_s (B, 1) [broadcastable to (B, K)],
        g_mu_raw (B, 1), g_sigma_raw (B, 1) -- the pre-tanh/pre-sigmoid raw
        MLP outputs, exposed only so `losses.safe_ttc_saturation_penalty`
        can discourage them from drifting into their nonlinearities'
        saturated tails (see that function's docstring for why this is
        needed: neither is directly supervised, so nothing else in the loss
        keeps them well-scaled). When a quantity is ablated
        (`mu_enabled`/`sigma_enabled=False`), its "_raw" output is a
        constant zero tensor -- harmless for the saturation penalty (a
        fixed, un-trained value never drifts toward the saturated tails
        the penalty guards against) and never touched by the optimizer
        since `self.g_mu`/`self.g_sigma` do not exist in that case.
        """
        B = h.shape[0]
        if self.mu_enabled:
            g_mu_raw = self.g_mu(h)                                    # (B, 1)
            correction = self.delta_max * torch.tanh(g_mu_raw)          # (B, 1)
        else:
            g_mu_raw = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
            correction = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
        mu_s = torch.log(tau_phy_seq.clamp_min(1e-6)) + correction    # (B, K)
        if self.sigma_enabled:
            g_sigma_raw = self.g_sigma(h)                              # (B, 1)
            sigma_s = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(g_sigma_raw)  # (B, 1)
        else:
            g_sigma_raw = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
            sigma_s = torch.full((B, 1), self._sigma_fixed, device=h.device, dtype=h.dtype)
        return mu_s, sigma_s, g_mu_raw, g_sigma_raw

    @staticmethod
    def lognormal_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-6)
        return _standard_normal_cdf((torch.log(x) - mu) / sigma)


class ResidualHead(nn.Module):
    """g_r, eq:bounded-residual. Disabled (forced to 0) during Stage 1
    (S3.8)."""

    def __init__(self, h_dim: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.mlp = _mlp(h_dim, hidden, 1, dropout=dropout)
        self.enabled = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        return self.mlp(h)


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p) - torch.log1p(-p)


class CARM(nn.Module):
    """Full per-agent Context-Adaptive Risk Model, composing S3.6.1-S3.6.5."""

    def __init__(
        self,
        d_kin: int,
        d_int: int,
        d_ctx: int,
        cfg: FutureRiskConfig = DEFAULT_CONFIG,
        hidden_dim: int = 64,
        rnn_hidden: int = 64,
        pooling: str = "attention",
        rnn_type: str = "gru",
        dropout: float = 0.1,
        safe_ttc_mu_enabled: bool = True,
        safe_ttc_sigma_enabled: bool = True,
    ):
        """`safe_ttc_mu_enabled`/`safe_ttc_sigma_enabled`: ablate
        eq:safettc's two learned quantities (Delta_mu, c_sigma)
        independently -- see `SafeTTCHead`'s docstring. Both default True
        (the full model); pass False to build one of the reduced
        full-R-decomposition ablation variants (root.tex
        Table~\\ref{tab:residual_ablation}'s extended rows: "raw" has both
        False, "+Delta_mu" has only sigma False, "+c_sigma" has only mu
        False, "Stage-1 only"/"Full CARM" have both True and differ only in
        `set_stage`/the residual)."""
        super().__init__()
        self.cfg = cfg
        self.encoder = InteractionEncoder(
            d_kin, d_int, d_ctx, hidden_dim=hidden_dim, rnn_hidden=rnn_hidden,
            pooling=pooling, rnn_type=rnn_type, dropout=dropout,
        )
        self.hazard_head = EventTimeHazardHead(self.encoder.out_dim, cfg.num_bins, dropout=dropout)
        self.safe_ttc_head = SafeTTCHead(
            self.encoder.out_dim, cfg.delta_max, cfg.sigma_min, cfg.sigma_max, dropout=dropout,
            mu_enabled=safe_ttc_mu_enabled, sigma_enabled=safe_ttc_sigma_enabled,
        )
        self.residual_head = ResidualHead(self.encoder.out_dim, dropout=dropout)
        self.beta = cfg.beta_residual
        self.eps = cfg.composition_eps

    def set_stage(self, stage: int):
        """Stage 1: residual disabled (g_r == 0). Stage 2: residual enabled,
        encoder/hazard/safe-TTC heads frozen (S3.8)."""
        if stage == 1:
            self.residual_head.enabled = False
            for p in self.parameters():
                p.requires_grad_(True)
        elif stage == 2:
            self.residual_head.enabled = True
            for module in (self.encoder, self.hazard_head, self.safe_ttc_head):
                for p in module.parameters():
                    p.requires_grad_(False)
            for p in self.residual_head.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError("stage must be 1 or 2")

    def forward(
        self,
        x_kin: torch.Tensor,          # (B, d_kin)
        X_int: torch.Tensor,          # (B, K, d_int)
        x_ctx: torch.Tensor,          # (B, d_ctx)
        tau_phy_seq: torch.Tensor,    # (B, K), tau_phy^j(b_k)
        bin_edges: torch.Tensor,      # (K+1,), b_0..b_K
        seq_mask: Optional[torch.Tensor] = None,
    ):
        """Returns a dict with all intermediate quantities needed for
        losses/inspection: h, q, p, F (hazard), mu_s, sigma_s, R_tilde
        (eq:composition), R (per-agent risk, eq:bounded-residual)."""
        h = self.encoder(x_kin, X_int, x_ctx, seq_mask=seq_mask)         # (B, hdim)
        q, p, Fk = self.hazard_head(h)                                   # (B, K) each
        mu_s, sigma_s, g_mu_raw, g_sigma_raw = self.safe_ttc_head(h, tau_phy_seq)  # (B,K),(B,1),(B,1),(B,1)

        b_km1 = bin_edges[:-1].unsqueeze(0)   # (1, K) -> b_0..b_{K-1}
        b_k = bin_edges[1:].unsqueeze(0)      # (1, K) -> b_1..b_K

        cdf_bk = self.safe_ttc_head.lognormal_cdf(b_k.expand_as(mu_s), mu_s, sigma_s)
        cdf_bkm1 = self.safe_ttc_head.lognormal_cdf(b_km1.expand_as(mu_s), mu_s, sigma_s)
        pr_interval = (cdf_bk - cdf_bkm1).clamp_min(0.0)                  # Pr(b_{k-1} < T_s <= b_k), (B,K)
        pr_tail = (1.0 - cdf_bk[:, -1:]).clamp_min(0.0)                    # Pr(T_s > b_K), (B,1), last bin's params

        # eq:composition
        R_tilde = (Fk * pr_interval).sum(dim=1, keepdim=True) + Fk[:, -1:] * pr_tail
        R_tilde = R_tilde.clamp(self.eps, 1 - self.eps)                   # (B, 1)

        # eq:bounded-residual
        residual = self.residual_head(h)                                  # (B, 1)
        R = torch.sigmoid(_logit(R_tilde) + self.beta * torch.tanh(residual))  # (B, 1)

        return {
            "h": h, "q": q, "p": p, "F": Fk,
            "mu_s": mu_s, "sigma_s": sigma_s,
            "g_mu_raw": g_mu_raw.squeeze(-1), "g_sigma_raw": g_sigma_raw.squeeze(-1),
            "R_tilde": R_tilde.squeeze(-1), "R": R.squeeze(-1),
        }


def aggregate_instance_risk(R_per_agent: torch.Tensor, agent_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """eq:agg: R = 1 - prod_j (1 - R^j), aggregated only across agents (no
    mode dimension -- single observed future).

    R_per_agent: (B, J) risk per agent per trajectory instance.
    agent_mask: (B, J) bool, True where agent j is real (False = padding).
    Padded slots are treated as R=0 (i.e. (1-0)=1, no contribution).
    """
    if agent_mask is not None:
        R_per_agent = torch.where(agent_mask, R_per_agent, torch.zeros_like(R_per_agent))
    return 1.0 - torch.prod(1.0 - R_per_agent, dim=-1)
