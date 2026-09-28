"""Two-stage CARM training procedure (S3.8).

Stage 1 (structured branch): residual disabled (g_r == 0), train encoder +
hazard head + safe-TTC head jointly with L = lambda_evt*L_evt +
lambda_risk*L_risk on the pooled nuPlan + Bench2Drive training set.

Stage 2 (bounded residual): freeze everything except g_r (and beta if made
learnable), train with L = lambda_risk*L_risk + lambda_rank*L_rank, using
L_rank only on Bench2Drive matched-pair data.

This module provides the orchestration logic; it expects a dataloader that
yields batches already carrying the preprocessed per-instance/per-agent
tensors produced by `data_pipeline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from .carm_model import CARM, aggregate_instance_risk
from .config import FutureRiskConfig, DEFAULT_CONFIG
from .losses import analytic_risk_loss, event_hazard_loss, rank_loss, safe_ttc_saturation_penalty, total_loss


@dataclass
class Batch:
    """Expected batch layout (produced by the data pipeline / a Dataset's
    collate_fn). Shapes:

    x_kin:        (B, J, d_kin)
    X_int:        (B, J, K, d_int)
    x_ctx:        (B, J, d_ctx)
    tau_phy_seq:  (B, J, K)
    agent_mask:   (B, J) bool, True = real agent
    T_evt:        (B, J) float, +inf if no event
    y_analytic:   (B,) float in [0,1], eq:phi target (per trajectory instance)
    pos_weight:   float, dataset-specific inverse-frequency weight for L_evt
                  (per-agent event rate)
    risk_pos_weight: float, analogous inverse-frequency weight for L_risk
                  (instance-level y_analytic>0.5 rate -- see
                  losses.analytic_risk_loss's docstring for why L_risk needs
                  its own weight distinct from L_evt's)
    bin_edges:    (K+1,) shared across the batch
    rank_pairs:   optional list of (worse_idx, better_idx) index pairs into
                  the batch, only present for Bench2Drive matched-pair data
    """
    x_kin: torch.Tensor
    X_int: torch.Tensor
    x_ctx: torch.Tensor
    tau_phy_seq: torch.Tensor
    agent_mask: torch.Tensor
    T_evt: torch.Tensor
    y_analytic: torch.Tensor
    bin_edges: torch.Tensor
    pos_weight: float = 1.0
    risk_pos_weight: float = 1.0
    rank_pairs: Optional[list] = None

    def to(self, device) -> "Batch":
        """Move every tensor field to `device` (e.g. 'cuda'), leaving
        `pos_weight`/`risk_pos_weight`/`rank_pairs` untouched. Extra
        bookkeeping attributes a collate_fn may have attached (e.g.
        dataset.py's `instance_ids`, `collision_label`) are not part of
        this dataclass's fields and are each dropped on the returned copy
        except `collision_label`, which is also moved if present since
        it's a tensor used in eval loops."""
        moved = Batch(
            x_kin=self.x_kin.to(device), X_int=self.X_int.to(device), x_ctx=self.x_ctx.to(device),
            tau_phy_seq=self.tau_phy_seq.to(device), agent_mask=self.agent_mask.to(device),
            T_evt=self.T_evt.to(device), y_analytic=self.y_analytic.to(device),
            bin_edges=self.bin_edges.to(device), pos_weight=self.pos_weight,
            risk_pos_weight=self.risk_pos_weight, rank_pairs=self.rank_pairs,
        )
        if hasattr(self, "collision_label"):
            moved.collision_label = self.collision_label.to(device)
        if hasattr(self, "instance_ids"):
            moved.instance_ids = self.instance_ids
        return moved


def _run_model_per_agent(model: CARM, batch: Batch):
    """Flatten the (B, J, ...) agent axis into the model's (N, ...) batch
    axis, run CARM once, then reshape back to (B, J)."""
    B, J = batch.agent_mask.shape
    flat = lambda t: t.reshape(B * J, *t.shape[2:])

    out = model(
        x_kin=flat(batch.x_kin),
        X_int=flat(batch.X_int),
        x_ctx=flat(batch.x_ctx),
        tau_phy_seq=flat(batch.tau_phy_seq),
        bin_edges=batch.bin_edges,
    )
    q = out["q"].view(B, J, -1)
    R_per_agent = out["R"].view(B, J)
    g_mu_raw = out["g_mu_raw"].view(B, J)
    g_sigma_raw = out["g_sigma_raw"].view(B, J)
    return q, R_per_agent, g_mu_raw, g_sigma_raw


def stage1_step(model: CARM, batch: Batch, cfg: FutureRiskConfig = DEFAULT_CONFIG) -> dict:
    """One Stage-1 forward + loss computation (residual disabled by the
    caller via model.set_stage(1))."""
    B, J = batch.agent_mask.shape
    q, R_per_agent, g_mu_raw, g_sigma_raw = _run_model_per_agent(model, batch)

    q_flat = q.reshape(B * J, -1)
    T_evt_flat = batch.T_evt.reshape(B * J)
    mask_flat = batch.agent_mask.reshape(B * J)
    L_evt = event_hazard_loss(
        q_flat, T_evt_flat, batch.bin_edges,
        pos_weight=batch.pos_weight, agent_mask=mask_flat,
    )

    R_instance = aggregate_instance_risk(R_per_agent, agent_mask=batch.agent_mask)  # eq:agg
    L_risk = analytic_risk_loss(R_instance, batch.y_analytic, pos_weight=batch.risk_pos_weight)

    # Not in the original spec -- see safe_ttc_saturation_penalty's docstring.
    L_saturation = safe_ttc_saturation_penalty(
        g_mu_raw.reshape(B * J), g_sigma_raw.reshape(B * J), agent_mask=mask_flat,
    )

    L = total_loss(L_evt=L_evt, L_risk=L_risk, L_saturation=L_saturation, cfg=cfg)
    return {
        "loss": L, "L_evt": L_evt.detach(), "L_risk": L_risk.detach(),
        "L_saturation": L_saturation.detach(), "R_instance": R_instance.detach(),
    }


def stage2_step(model: CARM, batch: Batch, cfg: FutureRiskConfig = DEFAULT_CONFIG) -> dict:
    """One Stage-2 forward + loss computation (encoder/hazard/safe-TTC
    frozen, only g_r trains, via model.set_stage(2))."""
    _, R_per_agent, _, _ = _run_model_per_agent(model, batch)
    R_instance = aggregate_instance_risk(R_per_agent, agent_mask=batch.agent_mask)
    L_risk = analytic_risk_loss(R_instance, batch.y_analytic, pos_weight=batch.risk_pos_weight)

    L_rank = None
    if batch.rank_pairs:
        worse = torch.stack([R_instance[w] for w, _ in batch.rank_pairs])
        better = torch.stack([R_instance[b] for _, b in batch.rank_pairs])
        L_rank = rank_loss(worse, better, gamma=cfg.rank_margin_gamma)

    L = total_loss(L_risk=L_risk, L_rank=L_rank, cfg=cfg)
    out = {"loss": L, "L_risk": L_risk.detach(), "R_instance": R_instance.detach()}
    if L_rank is not None:
        out["L_rank"] = L_rank.detach()
    return out


def train_stage(
    model: CARM,
    dataloader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    stage: int,
    num_epochs: int,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    on_epoch_end: Optional[Callable[[int, dict], None]] = None,
) -> None:
    """Runs one training stage end-to-end. `stage` in {1, 2}."""
    model.set_stage(stage)
    step_fn = stage1_step if stage == 1 else stage2_step

    for epoch in range(num_epochs):
        epoch_stats = {}
        n_batches = 0
        for batch in dataloader:
            optimizer.zero_grad()
            out = step_fn(model, batch, cfg=cfg)
            out["loss"].backward()
            optimizer.step()
            n_batches += 1
            for k, v in out.items():
                if k == "loss":
                    continue
                if torch.is_tensor(v) and v.numel() == 1:
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + v.item()
        epoch_stats = {k: v / max(n_batches, 1) for k, v in epoch_stats.items() if k != "R_instance"}
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_stats)


def run_two_stage_training(
    model: CARM,
    stage1_loader: Iterable[Batch],
    stage2_loader: Iterable[Batch],
    stage1_epochs: int,
    stage2_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    on_epoch_end: Optional[Callable[[int, dict], None]] = None,
) -> CARM:
    """Full Algorithm "CARM Two-Stage Training" (S3.8).

    weight_decay: L2 regularization on both stages' optimizers -- CARM has
    no other regularization besides this and the encoder/heads' own dropout
    (see carm_model.CARM's `dropout` arg), so leaving this at 0 on a small
    training set tends to show up as a visible train/val fit gap.
    """
    # Stage 1
    model.set_stage(1)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    train_stage(model, stage1_loader, opt1, stage=1, num_epochs=stage1_epochs, cfg=cfg, on_epoch_end=on_epoch_end)

    # Stage 2
    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    train_stage(model, stage2_loader, opt2, stage=2, num_epochs=stage2_epochs, cfg=cfg, on_epoch_end=on_epoch_end)

    return model
