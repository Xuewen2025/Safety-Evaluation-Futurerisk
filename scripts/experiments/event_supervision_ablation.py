#!/usr/bin/env python3
"""Critical-event supervision ablation ("Full CARM w/o L_evt" row): does the
independently supervised critical-event branch (Eq. l_evt) carry information
beyond what L_risk alone teaches the event-hazard head through the
composition (Eq. carm_composition)?

Trains a second full CARM model, IDENTICAL in every respect to the existing
v1 checkpoint (dataset/checkpoints/carm_deepaccident_v1.pt: same clip-level
split, seed=0, V1_CFG horizon_s=4.0/num_bins=20, width 64, dropout 0.1, Adam
lr=1e-3/wd=1e-4, batch 32, 40 Stage-1 + 20 Stage-2 epochs, native-target
override) except lambda_evt=0 for Stage 1. Stage 2 already excludes L_evt by
construction (training.stage2_step never computes it), so Stage 2 is
byte-for-byte the same procedure for both models.

The event-hazard head (EventTimeHazardHead: q_k/p_k/F_k) is NOT removed or
disabled: it still feeds eq:carm_composition, so it keeps receiving gradient
through L_risk alone. Setting lambda_evt=0 only zeroes the DIRECT
T_evt-supervised term -- verified numerically in `verify_gradient_isolation`
below before any training happens.

Reuses train_deepaccident_v1.py's exact protocol end-to-end (set_seed,
load_all_preprocessed, apply_native_target_override, evaluate_split,
predict_split_detailed, BestCheckpointTracker) and v1_full_eval.py's
fine-grained correlation protocol (severity_proxies/spearman_block on the
pooled train+val+test DeepAccident set, collision-free by the native label)
so every number is directly comparable to the existing
Table~residual_ablation / Table~finegrained_baselines rows.

Usage: python -m scripts.experiments.event_supervision_ablation
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.dataset_v1 import CARMDatasetV1, compute_pos_weight_v1, compute_risk_pos_weight_v1, make_collate_fn_v1
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import D_CTX_V1, D_EGO, D_OBJ_SEQ, build_instance_batch_item_v1
from futurerisk.safety_prior import analytic_trajectory_risk, context_dependent_margin
from futurerisk.split import split_instances_by_clip
from futurerisk.training import stage1_step, stage2_step, total_loss, event_hazard_loss, analytic_risk_loss

from scripts.train_deepaccident_v1 import (
    V1_CFG, STAGE1_EPOCHS, STAGE2_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, DROPOUT, SEED, DEVICE,
    DEEPACCIDENT_PREPROCESSED, NATIVE_LABELS_PATH, CHECKPOINT_PATH as FULL_CHECKPOINT_PATH,
    BestCheckpointTracker, apply_native_target_override, load_all_preprocessed,
    evaluate_split, predict_split_detailed, save_csv, set_seed,
)
from scripts.evaluate_crossdomain_v1_to_bench2drive import predict_R as predict_R_crossdomain

RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
CHECKPOINT_DIR = Path("dataset/checkpoints")
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
NO_EVT_CHECKPOINT_PATH = CHECKPOINT_DIR / "carm_deepaccident_v1_no_evt.pt"
FULL_MODEL_SUMMARY_PATH = RESULTS_DIR / "summary.json"

log = logging.getLogger("event_supervision_ablation")


def setup_logging(log_path: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


# --------------------------------------------------------------------------
# Required implementation check #2: lambda_e=0 removes ALL direct gradient
# from L_evt, while L_risk alone still drives the hazard head (check #3).
#
# CAVEAT (found while writing this check, pre-existing in the shared loss
# code, not introduced by this ablation): `analytic_risk_loss` clamps R to
# [1e-6, 1-1e-6] before BCE, but `aggregate_instance_risk`'s noisy-OR
# (1 - prod(1-R_j)) has no internal clamp. When several agents in one
# instance have moderately high per-agent R, the float32 product
# underflows to exactly 0.0, so R_instance lands exactly on the clamp's
# upper boundary -- and torch.clamp's gradient is exactly zero there. This
# means L_risk's gradient into the ENTIRE shared trunk (encoder, hazard
# head, safe-TTC head) is exactly zero for any batch where every instance
# saturates this way. In the full model this is harmless: L_evt supervises
# the hazard head through a separate, non-saturating per-bin BCE. In this
# ablation (lambda_evt=0), L_risk is the hazard head's ONLY gradient
# source, so how often batches saturate this way materially affects how
# much the ablated hazard head actually gets to learn -- `measure_
# saturation_frequency` below quantifies this over a full epoch so it can
# be reported as a caveat rather than silently affecting the result.
# --------------------------------------------------------------------------

def verify_gradient_isolation(train_loader) -> None:
    set_seed(SEED)
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=V1_CFG,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
    ).to(DEVICE)
    model.set_stage(1)
    batch = next(iter(train_loader)).to(DEVICE)

    B, J = batch.agent_mask.shape
    flat = lambda t: t.reshape(B * J, *t.shape[2:])
    out = model(
        x_kin=flat(batch.x_kin), X_int=flat(batch.X_int), x_ctx=flat(batch.x_ctx),
        tau_phy_seq=flat(batch.tau_phy_seq), bin_edges=batch.bin_edges,
    )
    q_flat = out["q"]
    T_evt_flat = batch.T_evt.reshape(B * J)
    mask_flat = batch.agent_mask.reshape(B * J)
    L_evt = event_hazard_loss(q_flat, T_evt_flat, batch.bin_edges, pos_weight=batch.pos_weight, agent_mask=mask_flat)
    R_instance = aggregate_instance_risk(out["R"].view(B, J), agent_mask=batch.agent_mask)
    L_risk = analytic_risk_loss(R_instance, batch.y_analytic, pos_weight=batch.risk_pos_weight)

    ablated_cfg = dataclasses.replace(V1_CFG, lambda_evt=0.0)
    L_ablated = total_loss(L_evt=L_evt, L_risk=L_risk, cfg=ablated_cfg)
    model.zero_grad()
    L_ablated.backward(retain_graph=True)
    hazard_grad_ablated = model.hazard_head.mlp[-1].weight.grad.clone()

    model.zero_grad()
    L_risk.backward()
    hazard_grad_risk_only = model.hazard_head.mlp[-1].weight.grad.clone()

    match = torch.allclose(hazard_grad_ablated, hazard_grad_risk_only, atol=1e-12)
    risk_nonzero = hazard_grad_risk_only.abs().sum().item() > 0.0
    log.info(f"[check] lambda_evt=0 hazard_head grad == L_risk-only hazard_head grad: {match}"
             f" (max abs diff={float((hazard_grad_ablated - hazard_grad_risk_only).abs().max()):.3e})")
    log.info(f"[check] hazard_head receives nonzero gradient via L_risk alone on batch 0: {risk_nonzero}"
             f" (grad abs-sum={float(hazard_grad_risk_only.abs().sum()):.4f}, R_instance sample={R_instance[:5].tolist()})")
    assert match, "lambda_evt=0 did not remove the direct L_evt gradient contribution to hazard_head"
    if not risk_nonzero:
        log.warning("      batch 0 hit the R_instance clamp-saturation issue (see module docstring) --"
                     " NOT treated as fatal here; measure_saturation_frequency() below quantifies how"
                     " often this happens across a full epoch.")


def measure_saturation_frequency(train_loader, ablated_cfg) -> dict:
    """Fraction of Stage-1 training batches (one full epoch, freshly
    initialized model, ablated cfg) where L_risk's gradient into
    hazard_head is exactly zero -- i.e. every instance in that batch hit
    the R_instance clamp-saturation boundary described above. Diagnostic
    only: does not affect training, which proceeds with whatever gradient
    actually flows, exactly as it would for any real training run."""
    set_seed(SEED)
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=ablated_cfg,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
    ).to(DEVICE)
    model.set_stage(1)

    n_batches, n_zero_grad_batches, n_instances, n_saturated_instances = 0, 0, 0, 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        out = stage1_step(model, batch, cfg=ablated_cfg)
        model.zero_grad()
        out["loss"].backward()
        grad_abs_sum = model.hazard_head.mlp[-1].weight.grad.abs().sum().item()
        n_batches += 1
        n_zero_grad_batches += int(grad_abs_sum == 0.0)
        n_instances += batch.y_analytic.shape[0]
        n_saturated_instances += int((out["R_instance"] >= 1.0 - 1e-6).sum().item())
    model.zero_grad()

    frac_batches = n_zero_grad_batches / max(n_batches, 1)
    frac_instances = n_saturated_instances / max(n_instances, 1)
    log.info(f"[check] saturation frequency over 1 full Stage-1 epoch ({n_batches} batches, {n_instances} instances,"
             f" freshly initialized model): {n_zero_grad_batches}/{n_batches} batches ({frac_batches:.1%}) had"
             f" EXACTLY ZERO hazard_head gradient; {n_saturated_instances}/{n_instances} instances ({frac_instances:.1%})"
             f" individually hit the R_instance>=1-1e-6 clamp boundary.")
    return {
        "n_batches": n_batches, "n_zero_grad_batches": n_zero_grad_batches, "frac_zero_grad_batches": frac_batches,
        "n_instances": n_instances, "n_saturated_instances": n_saturated_instances, "frac_saturated_instances": frac_instances,
    }


# --------------------------------------------------------------------------
# Train the w/o-event-supervision variant (full two-stage schedule)
# --------------------------------------------------------------------------

def train_no_evt_variant(loaders) -> CARM:
    train_loader, val_loader, _ = loaders
    ablated_cfg = dataclasses.replace(V1_CFG, lambda_evt=0.0)
    log.info(f"[train] lambda_evt={ablated_cfg.lambda_evt} lambda_risk={ablated_cfg.lambda_risk}"
             f" lambda_saturation={ablated_cfg.lambda_saturation} (Stage 2 always excludes L_evt/lambda_evt)")

    set_seed(SEED)  # identical init/data order to train_deepaccident_v1.py's full model
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=ablated_cfg,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
    ).to(DEVICE)

    best = BestCheckpointTracker()
    model.set_stage(1)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 1 (L_risk + L_saturation only; L_evt weight is 0):")
    for epoch in range(STAGE1_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt1.zero_grad()
            out = stage1_step(model, batch, cfg=ablated_cfg)
            out["loss"].backward()
            opt1.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, ablated_cfg)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        stage1 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 2 (L_risk only -- identical procedure to the full model):")
    for epoch in range(STAGE2_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt2.zero_grad()
            out = stage2_step(model, batch, cfg=ablated_cfg)
            out["loss"].backward()
            opt2.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, ablated_cfg)
        is_best = best.maybe_update(model, val_metrics, stage=2, epoch=epoch)
        log.info(f"        stage2 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    log.info(f"[train] best checkpoint: stage={best.stage} epoch={best.epoch}"
             f" val_auroc={best.best_auroc:.3f} val_auprc={best.best_auprc:.3f} -- restoring it")
    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})
    return model, ablated_cfg, best


# --------------------------------------------------------------------------
# Fine-grained correlation: rho(R, a_max) and rho(R, -m), same protocol as
# v1_full_eval.py's finegrained_baselines table (pooled train+val+test,
# native-label collision-free subset).
# --------------------------------------------------------------------------

def score_finegrained_row(model, cfg, pre) -> dict:
    raw = raw_instance_from_preprocessed(pre)
    item = build_instance_batch_item_v1(raw, pre, cfg)
    if item["x_ego"].shape[0] == 0:
        R_pred = 0.0
    else:
        x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
        X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
        x_ctx = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
        tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
        bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
        R_pred = float(aggregate_instance_risk(out["R"].unsqueeze(0)).item())

    if not pre.tau_3d:
        worst_margin, max_sev = float("inf"), 0.0
    else:
        margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
        worst_margin = float(np.min(np.concatenate(margins))) if margins else float("inf")
        max_sev = max(float(np.max(pre.evasive_severity[a])) for a in pre.evasive_severity) if pre.evasive_severity else 0.0
    return {"R_predicted": R_pred, "worst_margin": worst_margin, "max_evasive_severity": max_sev}


def finegrained_correlations(model, cfg, split, native_labels) -> dict:
    """rho(R, a_max) and rho(R, -m) on the pooled train+val+test DeepAccident
    set, restricted to native-label collision-free instances -- identical
    subset/sign convention to v1_full_eval.py's spearman_block."""
    rows = []
    for part in ("train", "val", "test"):
        for iid in split[part]:
            p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
            if not p.exists():
                continue
            pre = load_preprocessed_instance(str(p))
            if pre.ego_box is None:
                continue
            entry = native_labels.get(iid)
            if entry is None or entry.get("collision_label") is None:
                continue
            row = score_finegrained_row(model, cfg, pre)
            row["native_collision_label"] = int(entry["collision_label"])
            rows.append(row)

    free = [r for r in rows if r["native_collision_label"] == 0]
    out = {}
    for proxy, sign in (("max_evasive_severity", 1.0), ("worst_margin", -1.0)):
        vals = np.array([r[proxy] for r in free])
        finite = np.isfinite(vals)
        scores = np.array([r["R_predicted"] for r in free])[finite]
        v = vals[finite]
        if len(v) < 10 or np.all(v == v[0]):
            out[proxy] = None
            continue
        rho, p = spearmanr(scores, v)
        out[proxy] = {"n": int(len(v)), "spearman_rho": float(sign * rho), "p_value": float(p)}
    return out


# --------------------------------------------------------------------------
# Zero-shot Bench2Drive cross-domain eval (same protocol as
# evaluate_crossdomain_v1_to_bench2drive.py).
# --------------------------------------------------------------------------

@torch.no_grad()
def crossdomain_bench2drive(model, cfg) -> dict:
    R_all, lbl_all = [], []
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        R_all.append(predict_R_crossdomain(model, cfg, raw, pre))
        lbl_all.append(int(any(f.any() for f in raw.agent_collision_flags.values())))
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} Bench2Drive instances scored")
    R_all, lbl_all = np.array(R_all), np.array(lbl_all)
    return discrimination_metrics(R_all, lbl_all)


# --------------------------------------------------------------------------
# Optional event-head diagnostic: agent-level critical-event AUROC/AUPRC
# (F_K = total event probability mass within the horizon, vs. whether
# T_evt is finite), plus mean predicted event probability by trigger status.
# Computed on the DeepAccident test split, for both models.
# --------------------------------------------------------------------------

@torch.no_grad()
def event_head_diagnostic(model, cfg, test_loader) -> dict:
    model.eval()
    F_last_all, triggered_all = [], []
    for batch in test_loader:
        batch = batch.to(DEVICE)
        B, J = batch.agent_mask.shape
        flat = lambda t: t.reshape(B * J, *t.shape[2:])
        out = model(
            x_kin=flat(batch.x_kin), X_int=flat(batch.X_int), x_ctx=flat(batch.x_ctx),
            tau_phy_seq=flat(batch.tau_phy_seq), bin_edges=batch.bin_edges,
        )
        F_last = out["F"][:, -1].view(B, J)          # total event prob. mass within horizon, per agent
        mask = batch.agent_mask
        triggered = torch.isfinite(batch.T_evt)
        F_last_all.append(F_last[mask].cpu().numpy())
        triggered_all.append(triggered[mask].cpu().numpy())
    model.train()
    F_last_all = np.concatenate(F_last_all)
    triggered_all = np.concatenate(triggered_all).astype(int)
    m = discrimination_metrics(F_last_all, triggered_all)
    mean_triggered = float(F_last_all[triggered_all == 1].mean()) if (triggered_all == 1).any() else float("nan")
    mean_not = float(F_last_all[triggered_all == 0].mean()) if (triggered_all == 0).any() else float("nan")
    return {
        "agent_level_event_auroc": m["auroc"], "agent_level_event_auprc": m["auprc"],
        "n_agents": int(len(F_last_all)), "n_triggered": int(triggered_all.sum()),
        "mean_F_last_triggered": mean_triggered, "mean_F_last_not_triggered": mean_not,
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOGS_DIR / f"event_supervision_ablation_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info("[0/5] Loading cached DeepAccident preprocessing (identical to train_deepaccident_v1.py) ...")
    da_pre = load_all_preprocessed(DEEPACCIDENT_PREPROCESSED)
    split = split_instances_by_clip(list(da_pre.keys()), train_frac=0.6, val_frac=0.2, seed=SEED)
    raw_by_id = {iid: raw_instance_from_preprocessed(pre) for iid, pre in da_pre.items()}
    datasets = {part: CARMDatasetV1([raw_by_id[i] for i in ids], da_pre, cfg=V1_CFG) for part, ids in split.items()}
    n_overridden = apply_native_target_override(datasets, native_labels)
    log.info(f"      native-ground-truth target override: {n_overridden} instances")

    pos_weight = compute_pos_weight_v1(datasets["train"])
    risk_pos_weight = compute_risk_pos_weight_v1(datasets["train"])
    collate = make_collate_fn_v1(cfg=V1_CFG, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False, collate_fn=collate)
    loaders = (train_loader, val_loader, test_loader)

    log.info("[1/5] Required check #2/#3: verifying lambda_evt=0 gradient isolation ...")
    verify_gradient_isolation(train_loader)
    ablated_cfg_for_diag = dataclasses.replace(V1_CFG, lambda_evt=0.0)
    saturation_diag = measure_saturation_frequency(train_loader, ablated_cfg_for_diag)

    log.info("[2/5] Required check #1: full-model reference (existing checkpoint/summary) ...")
    if not FULL_MODEL_SUMMARY_PATH.exists():
        log.warning(f"      {FULL_MODEL_SUMMARY_PATH} not found -- run train_deepaccident_v1.py first;"
                    " proceeding without the full-model in-domain reference row.")
        full_model_native = None
    else:
        with open(FULL_MODEL_SUMMARY_PATH) as f:
            full_summary = json.load(f)
        full_model_native = full_summary["native"]
        log.info(f"      Full CARM (existing checkpoint) in-domain native AUROC={full_model_native['auroc']:.3f}"
                 f" AUPRC={full_model_native['auprc']:.3f}  (root.tex reports 0.793 / 0.490)")

    log.info("[3/5] Training Full CARM w/o L_evt (lambda_evt=0, full two-stage schedule) ...")
    model, ablated_cfg, best = train_no_evt_variant(loaders)

    test_rows = predict_split_detailed(model, test_loader, native_labels)
    save_csv(test_rows, RESULTS_DIR / "predictions_test_no_evt.csv")
    native_rows = [r for r in test_rows if r["collision_label_native"] is not None]
    R_nat = np.array([r["R_predicted"] for r in native_rows])
    lbl_nat = np.array([r["collision_label_native"] for r in native_rows])
    m_nat = discrimination_metrics(R_nat, lbl_nat)
    log.info(f"      DeepAccident in-domain test (native label) n={len(native_rows)}"
             f"  AUROC={m_nat['auroc']:.3f}  AUPRC={m_nat['auprc']:.3f}")

    ckpt_path = NO_EVT_CHECKPOINT_PATH
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict, "cfg": dataclasses.asdict(ablated_cfg),
        "d_ego": D_EGO, "d_obj": D_OBJ_SEQ, "d_ctx_v1": D_CTX_V1, "split": split,
        "best_val_auroc": best.best_auroc, "best_val_auprc": best.best_auprc,
    }, ckpt_path)
    log.info(f"      saved checkpoint to {ckpt_path}")

    log.info("[4/5] Zero-shot Bench2Drive cross-domain eval ...")
    m_cross = crossdomain_bench2drive(model, ablated_cfg)
    log.info(f"      Bench2Drive cross-domain (geometric label)  AUROC={m_cross['auroc']:.3f} AUPRC={m_cross['auprc']:.3f}")

    log.info("[4/5] Fine-grained correlation: rho(R, a_max) and rho(R, -m) ...")
    finegrained = finegrained_correlations(model, ablated_cfg, split, native_labels)
    for proxy, label in (("max_evasive_severity", "a_max"), ("worst_margin", "-m")):
        v = finegrained.get(proxy)
        if v:
            log.info(f"      rho(R, {label}) = {v['spearman_rho']:+.3f} (n={v['n']}, p={v['p_value']:.2e})")
        else:
            log.info(f"      rho(R, {label}) = NA")

    log.info("[5/5] Optional event-head diagnostic (agent-level F_K vs. T_evt-finite) ...")
    event_diag = event_head_diagnostic(model, ablated_cfg, test_loader)
    log.info(f"      no-evt model: agent-level event AUROC={event_diag['agent_level_event_auroc']:.3f}"
             f" AUPRC={event_diag['agent_level_event_auprc']:.3f}"
             f" (n={event_diag['n_agents']}, triggered={event_diag['n_triggered']})"
             f"  mean F_K: triggered={event_diag['mean_F_last_triggered']:.3f}"
             f" not-triggered={event_diag['mean_F_last_not_triggered']:.3f}")

    full_model_event_diag = None
    if FULL_CHECKPOINT_PATH and Path(FULL_CHECKPOINT_PATH).exists():
        from futurerisk.config import FutureRiskConfig
        full_ckpt = torch.load(FULL_CHECKPOINT_PATH, map_location=DEVICE)
        full_cfg = FutureRiskConfig(**{k: v for k, v in full_ckpt["cfg"].items() if k != "physical_prior_by_class"})
        full_model = CARM(
            d_kin=full_ckpt["d_ego"], d_int=full_ckpt["d_obj"], d_ctx=full_ckpt["d_ctx_v1"], cfg=full_cfg,
            hidden_dim=64, rnn_hidden=64, dropout=0.1,
        ).to(DEVICE)
        full_model.load_state_dict(full_ckpt["model_state_dict"])
        full_model_event_diag = event_head_diagnostic(full_model, full_cfg, test_loader)
        log.info(f"      full model:  agent-level event AUROC={full_model_event_diag['agent_level_event_auroc']:.3f}"
                 f" AUPRC={full_model_event_diag['agent_level_event_auprc']:.3f}"
                 f"  mean F_K: triggered={full_model_event_diag['mean_F_last_triggered']:.3f}"
                 f" not-triggered={full_model_event_diag['mean_F_last_not_triggered']:.3f}")

    results = {
        "full_carm_reference": full_model_native,
        "saturation_diagnostic": saturation_diag,
        "no_evt": {
            "auroc_indomain": m_nat["auroc"], "auprc_indomain": m_nat["auprc"],
            "auroc_crossdomain": m_cross["auroc"], "auprc_crossdomain": m_cross["auprc"],
            "finegrained_rho": finegrained,
            "event_head_diagnostic": event_diag,
        },
        "full_carm_event_head_diagnostic": full_model_event_diag,
    }
    with open(RESULTS_DIR / "event_supervision_ablation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {RESULTS_DIR / 'event_supervision_ablation_summary.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
