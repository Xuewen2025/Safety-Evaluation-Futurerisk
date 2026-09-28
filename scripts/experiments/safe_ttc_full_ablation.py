#!/usr/bin/env python3
"""Full-R decomposition ablation (root.tex Table~\\ref{tab:residual_ablation},
extended from 2 rows to 5): every term that composes R^j (eqs. safettc,
composition, bounded-residual) ablated, one nested addition at a time,
evaluated on BOTH DeepAccident (in-domain, native label) and Bench2Drive
(cross-domain, zero-shot, geometric label):

  1. raw            mu_s = log(tau_phy) exactly (Delta_mu off), sigma_s
                     fixed at the midpoint (sigma_min+sigma_max)/2 (c_sigma
                     off), no residual. Only the encoder + hazard head
                     (q_k/F_k, eq:hazard) are learned; the safe-TTC head
                     contributes nothing beyond the closed-form physical
                     prior itself.
  2. +Delta_mu       mu_s learned (Delta_mu on), sigma_s still fixed
                     (c_sigma off), no residual.
  3. +c_sigma        mu_s = log(tau_phy) exactly (Delta_mu off), sigma_s
                     learned (c_sigma on), no residual.
  4. Stage-1 only    both Delta_mu and c_sigma learned, no residual --
                     this is the SAME checkpoint/numbers as root.tex's
                     existing "Stage-1 only (g_r=0)" row (dataset/results_v1/
                     v1_interpretability_summary.json), reused here rather
                     than retrained, since it needs no new checkpoint.
  5. Full CARM       both Delta_mu and c_sigma learned, plus the Stage-2
                     bounded residual -- same source as row 4's, the
                     existing "Full CARM" row.

Rows 1-3 require new checkpoints (SafeTTCHead's mu_enabled/sigma_enabled
ablate architecture, not just a frozen/zeroed forward pass, so the encoder
+ hazard head must be retrained jointly with whichever half of the
safe-TTC head is present -- see `futurerisk.carm_model.SafeTTCHead`'s
docstring). Rows 4-5 need no new training or evaluation run at all: they
are read directly from the existing `v1_interpretability_summary.json`
(itself inference-only on the one existing Full-CARM checkpoint, since
Stage-1-only vs Full CARM there is just R_tilde vs R from that one
checkpoint's forward pass -- no architectural difference).

Same protocol throughout as `train_deepaccident_v1.py` (identical seed,
split, hyperparameters, native-ground-truth target override) and
`evaluate_crossdomain_v1_to_bench2drive.py` / `v1_interpretability.py`
(Bench2Drive: zero-shot, no fine-tuning, geometric label, since Bench2Drive
has no native annotation) -- only the ablation flags differ, so rows are
directly comparable to each other and to the existing Table III rows.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.split import split_instances_by_clip
from futurerisk.dataset_v1 import CARMDatasetV1, compute_pos_weight_v1, compute_risk_pos_weight_v1, make_collate_fn_v1
from futurerisk.training import stage1_step
from scripts.train_deepaccident_v1 import (
    V1_CFG, STAGE1_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, DROPOUT, SEED, DEVICE,
    DEEPACCIDENT_PREPROCESSED, NATIVE_LABELS_PATH,
    BestCheckpointTracker, apply_native_target_override, load_all_preprocessed,
    evaluate_split, predict_split_detailed, save_csv, set_seed,
)
from futurerisk.features_v1 import D_CTX_V1, D_EGO, D_OBJ_SEQ

BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
CHECKPOINT_DIR = Path("dataset/checkpoints")
EXISTING_STAGE12_SUMMARY = RESULTS_DIR / "v1_interpretability_summary.json"

# (variant name, mu_enabled, sigma_enabled)
VARIANTS = [
    ("raw", False, False),
    ("delta_mu_only", True, False),
    ("c_sigma_only", False, True),
]

log = logging.getLogger("safe_ttc_full_ablation")


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


def train_one_variant(name: str, mu_enabled: bool, sigma_enabled: bool, split, loaders, native_labels) -> dict:
    train_loader, val_loader, test_loader = loaders
    set_seed(SEED)  # re-seed identically before each variant's model init, matching train_deepaccident_v1.py's protocol
    log.info(f"[variant={name}] mu_enabled={mu_enabled} sigma_enabled={sigma_enabled}")
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=V1_CFG,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
        safe_ttc_mu_enabled=mu_enabled, safe_ttc_sigma_enabled=sigma_enabled,
    ).to(DEVICE)
    log.info(f"      model params: {sum(p.numel() for p in model.parameters())}")

    best = BestCheckpointTracker()
    model.set_stage(1)  # residual stays disabled (g_r==0) throughout -- these are all no-residual rows
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(STAGE1_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = stage1_step(model, batch, cfg=V1_CFG)
            out["loss"].backward()
            opt.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, V1_CFG)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        [{name}] epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    log.info(f"      [{name}] best: epoch={best.epoch} val_auroc={best.best_auroc:.3f} -- restoring")
    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})

    test_rows = predict_split_detailed(model, test_loader, native_labels)
    save_csv(test_rows, RESULTS_DIR / f"predictions_test_safettc_{name}.csv")
    native_rows = [r for r in test_rows if r["collision_label_native"] is not None]
    R_nat = np.array([r["R_predicted"] for r in native_rows])
    lbl_nat = np.array([r["collision_label_native"] for r in native_rows])
    m_nat = discrimination_metrics(R_nat, lbl_nat) if len(native_rows) else {"auroc": float("nan"), "auprc": float("nan")}
    log.info(f"      [{name}] DeepAccident in-domain test (native label) n={len(native_rows)}"
             f"  AUROC={m_nat['auroc']:.3f}  AUPRC={m_nat['auprc']:.3f}")

    ckpt_path = CHECKPOINT_DIR / f"carm_deepaccident_v1_safettc_{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "cfg": dataclasses.asdict(V1_CFG),
        "d_ego": D_EGO, "d_obj": D_OBJ_SEQ, "d_ctx_v1": D_CTX_V1,
        "safe_ttc_mu_enabled": mu_enabled, "safe_ttc_sigma_enabled": sigma_enabled,
        "split": split,
        "best_val_auroc": best.best_auroc, "best_val_auprc": best.best_auprc,
    }, ckpt_path)
    log.info(f"      [{name}] saved checkpoint to {ckpt_path}")

    return {
        "auroc_indomain": m_nat["auroc"], "auprc_indomain": m_nat["auprc"],
        "_model_state_for_crossdomain": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "_mu_enabled": mu_enabled, "_sigma_enabled": sigma_enabled,
    }


@torch.no_grad()
def crossdomain_bench2drive(name: str, mu_enabled: bool, sigma_enabled: bool, state_dict: dict) -> dict:
    """Zero-shot Bench2Drive cross-domain eval, same protocol as
    `v1_interpretability.py`'s step [2/3] / `evaluate_crossdomain_v1_to_
    bench2drive.py`: every preprocessed Bench2Drive instance, geometric
    collision label (no native annotation exists for Bench2Drive)."""
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=V1_CFG,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
        safe_ttc_mu_enabled=mu_enabled, safe_ttc_sigma_enabled=sigma_enabled,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    R_all, lbl_all = [], []
    paths = sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))
    for i, p in enumerate(paths):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        item = build_instance_batch_item_v1(raw, pre, V1_CFG)
        if item["x_ego"].shape[0] == 0:
            continue
        x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
        X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
        x_ctx = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
        tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
        bin_edges = torch.tensor(V1_CFG.bin_edges, dtype=torch.float32, device=DEVICE)
        out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
        R_instance = aggregate_instance_risk(out["R"].unsqueeze(0))  # (1, J) -> (1,), matching predict_R's convention
        R_all.append(float(R_instance.item()))
        lbl_all.append(int(any(f.any() for f in raw.agent_collision_flags.values())))
        if (i + 1) % 1000 == 0:
            log.info(f"      [{name}] ... {i+1} Bench2Drive instances scored")

    R_all, lbl_all = np.array(R_all), np.array(lbl_all)
    m = discrimination_metrics(R_all, lbl_all)
    log.info(f"      [{name}] Bench2Drive cross-domain (geometric label) n={len(R_all)}"
             f"  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")
    return {"auroc_crossdomain": m["auroc"], "auprc_crossdomain": m["auprc"]}


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOGS_DIR / f"safe_ttc_full_ablation_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info("[setup] loading cached DeepAccident preprocessing (identical to train_deepaccident_v1.py) ...")
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

    rows = {}
    for name, mu_enabled, sigma_enabled in VARIANTS:
        r = train_one_variant(name, mu_enabled, sigma_enabled, split, loaders, native_labels)
        cross = crossdomain_bench2drive(name, mu_enabled, sigma_enabled, r.pop("_model_state_for_crossdomain"))
        r.update(cross)
        r.pop("_mu_enabled", None); r.pop("_sigma_enabled", None)
        rows[name] = r
        log.info(f"[done] {name}: AUROC(in)={r['auroc_indomain']:.3f} AUPRC(in)={r['auprc_indomain']:.3f}"
                 f"  AUROC(cross)={r['auroc_crossdomain']:.3f} AUPRC(cross)={r['auprc_crossdomain']:.3f}")

    if EXISTING_STAGE12_SUMMARY.exists():
        with open(EXISTING_STAGE12_SUMMARY) as f:
            existing = json.load(f)["residual_ablation"]
        rows["stage1_only"] = existing["stage1_only"]
        rows["full_carm"] = existing["full_carm"]
        log.info("[reused] stage1_only / full_carm rows read from v1_interpretability_summary.json (no retrain needed)")
    else:
        log.warning(f"      {EXISTING_STAGE12_SUMMARY} not found -- stage1_only/full_carm rows omitted;"
                    " run v1_interpretability.py first if you need the full 5-row table.")

    with open(RESULTS_DIR / "safe_ttc_full_ablation.json", "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"wrote {RESULTS_DIR / 'safe_ttc_full_ablation.json'}")

    order = ["raw", "delta_mu_only", "c_sigma_only", "stage1_only", "full_carm"]
    log.info("")
    log.info(f"{'variant':16s} {'AUROC(in)':>10s} {'AUPRC(in)':>10s} {'AUROC(cross)':>13s} {'AUPRC(cross)':>13s}")
    for name in order:
        if name not in rows:
            continue
        r = rows[name]
        log.info(f"{name:16s} {r['auroc_indomain']:>10.3f} {r['auprc_indomain']:>10.3f}"
                 f" {r['auroc_crossdomain']:>13.3f} {r['auprc_crossdomain']:>13.3f}")

    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
