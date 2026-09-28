#!/usr/bin/env python3
"""Ablation of `train_deepaccident_v1.py`: remove DIRECT critical-event
supervision (lambda_evt=0) while keeping the event/hazard head itself in
the architecture (it can still receive gradients indirectly through
L_risk, since R_instance's composition depends on it).

Everything else -- seed, clip-level split, horizon/bins, model width,
dropout, optimizer, LR, weight decay, batch size, epoch counts, Stage-2
freezing procedure, native-label target override -- is IDENTICAL to
`train_deepaccident_v1.py`; the only diff is V1_CFG.lambda_evt=0.0 and the
output paths (separate checkpoint/summary/log so this never touches the
existing Full CARM artifacts).

See icra/add_experiments.txt-style provenance note: this script exists to
answer "does independently-supervised critical-event timing help beyond
what L_risk alone would teach the hazard head", per explicit user request
for a w/o-L_evt row in the residual-ablation table.
"""
from __future__ import annotations

import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.dataset_v1 import CARMDatasetV1, compute_pos_weight_v1, compute_risk_pos_weight_v1, make_collate_fn_v1
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features_v1 import D_CTX_V1, D_EGO, D_OBJ_SEQ
from futurerisk.split import split_instances_by_clip
from futurerisk.training import stage1_step, stage2_step
from scripts.train_deepaccident_v1 import (
    apply_native_target_override, evaluate_split, load_all_preprocessed,
    predict_split_detailed, save_csv, set_seed, BestCheckpointTracker,
)

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1_no_event.pt"
LOGS_DIR = Path("dataset/logs")
RESULTS_DIR = Path("dataset/results_v1")

V1_CFG = FutureRiskConfig(horizon_s=4.0, num_bins=20, lambda_evt=0.0)  # ONLY diff vs. train_deepaccident_v1.py

STAGE1_EPOCHS = 40
STAGE2_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
SEED = 0  # same seed as the existing Full CARM checkpoint (single-seed protocol, matches every other row in this project)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("train_deepaccident_v1_no_event")


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


def main():
    set_seed(SEED)
    t0 = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_deepaccident_v1_no_event_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    log.info(f"v1 config (ABLATION: lambda_evt={V1_CFG.lambda_evt}): horizon_s={V1_CFG.horizon_s}, "
             f"num_bins={V1_CFG.num_bins} (bin width={V1_CFG.horizon_s/V1_CFG.num_bins:.2f}s), DeepAccident-only")
    assert V1_CFG.lambda_evt == 0.0, "this script only makes sense as the w/o-L_evt ablation"

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info(f"[1/6] Loading cached DeepAccident preprocessing from {DEEPACCIDENT_PREPROCESSED} ...")
    da_pre = load_all_preprocessed(DEEPACCIDENT_PREPROCESSED)
    log.info(f"      {len(da_pre)} instances")

    split = split_instances_by_clip(list(da_pre.keys()), train_frac=0.6, val_frac=0.2, seed=SEED)
    log.info(f"[2/6] Clip-level split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    raw_by_id = {iid: raw_instance_from_preprocessed(pre) for iid, pre in da_pre.items()}
    datasets = {
        part: CARMDatasetV1([raw_by_id[i] for i in ids], da_pre, cfg=V1_CFG)
        for part, ids in split.items()
    }
    for part, ds in datasets.items():
        log.info(f"      {part} dataset (>=1 agent): {len(ds)} instances")

    n_overridden = apply_native_target_override(datasets, native_labels)
    log.info(f"      native-ground-truth target override: {n_overridden} instances forced to"
             " y_analytic=1.0 (native says collision, geometric proxy said otherwise)")

    n_train_positive = sum(1 for item in datasets["train"].items if item["y_analytic"] > 0.5)
    log.info(f"      train instances with y_analytic > 0.5: {n_train_positive} / {len(datasets['train'])}")

    pos_weight = compute_pos_weight_v1(datasets["train"])
    risk_pos_weight = compute_risk_pos_weight_v1(datasets["train"])
    log.info(f"      pos_weight(L_evt)={pos_weight:.2f} (unused: lambda_evt=0)  risk_pos_weight(L_risk)={risk_pos_weight:.2f}")

    collate = make_collate_fn_v1(cfg=V1_CFG, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False, collate_fn=collate)

    log.info(f"[3/6] Building CARM (d_ego={D_EGO}, d_obj={D_OBJ_SEQ}, d_ctx_v1={D_CTX_V1}, dropout={DROPOUT})"
             f" on device={DEVICE} ...")
    model = CARM(
        d_kin=D_EGO, d_int=D_OBJ_SEQ, d_ctx=D_CTX_V1, cfg=V1_CFG,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
    ).to(DEVICE)
    log.info(f"      model params: {sum(p.numel() for p in model.parameters())}")

    best = BestCheckpointTracker()

    model.set_stage(1)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 1 (L_risk + L_saturation only; L_evt weight=0, event head still gets gradients via L_risk):")
    for epoch in range(STAGE1_EPOCHS):
        epoch_losses = []
        epoch_L_evt = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt1.zero_grad()
            out = stage1_step(model, batch, cfg=V1_CFG)
            out["loss"].backward()
            opt1.step()
            epoch_losses.append(out["loss"].item())
            epoch_L_evt.append(out["L_evt"].item())
        val_metrics = evaluate_split(model, val_loader, V1_CFG)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        stage1 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  raw_L_evt(unweighted,not-backpropped)={np.mean(epoch_L_evt):.4f}"
                 f"  val_L_risk={val_metrics['mean_L_risk']:.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"  (n={val_metrics['n_instances']}, collisions={val_metrics['n_collisions']})"
                 f"{'  *best*' if is_best else ''}")

    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 2 (L_risk only, same freezing as train_deepaccident_v1.py):")
    for epoch in range(STAGE2_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt2.zero_grad()
            out = stage2_step(model, batch, cfg=V1_CFG)
            out["loss"].backward()
            opt2.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, V1_CFG)
        is_best = best.maybe_update(model, val_metrics, stage=2, epoch=epoch)
        log.info(f"        stage2 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_L_risk={val_metrics['mean_L_risk']:.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    log.info(f"[4/6] Best checkpoint: stage={best.stage} epoch={best.epoch}"
             f" val_auroc(geometric)={best.best_auroc:.3f} val_auprc(geometric)={best.best_auprc:.3f} -- restoring it")
    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})

    log.info("[5/6] Test-set evaluation (both geometric and native labels) ...")
    test_rows = predict_split_detailed(model, test_loader, native_labels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(test_rows, RESULTS_DIR / "predictions_test_no_event.csv")

    R = np.array([r["R_predicted"] for r in test_rows])
    lbl_geo = np.array([r["collision_label_geometric"] for r in test_rows])
    native_rows = [r for r in test_rows if r["collision_label_native"] is not None]
    R_nat = np.array([r["R_predicted"] for r in native_rows])
    lbl_nat = np.array([r["collision_label_native"] for r in native_rows])

    m_geo = discrimination_metrics(R, lbl_geo)
    m_nat = discrimination_metrics(R_nat, lbl_nat) if len(native_rows) else {"auroc": float("nan"), "auprc": float("nan")}
    log.info(f"      test (geometric label)  n={len(test_rows):5d}  collisions={int(lbl_geo.sum()):5d}"
             f"  AUROC={m_geo['auroc']:.3f}  AUPRC={m_geo['auprc']:.3f}")
    log.info(f"      test (native label)     n={len(native_rows):5d}  collisions={int(lbl_nat.sum()):5d}"
             f"  AUROC={m_nat['auroc']:.3f}  AUPRC={m_nat['auprc']:.3f}")

    results = {
        "ablation": "lambda_evt=0.0 (no direct critical-event supervision)",
        "n_test": len(test_rows),
        "geometric": {"n": len(test_rows), "n_collisions": int(lbl_geo.sum()), **m_geo},
        "native": {"n": len(native_rows), "n_collisions": int(lbl_nat.sum()), **m_nat},
    }

    ckpt_path = Path(CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "cfg": dataclasses.asdict(V1_CFG),
        "d_ego": D_EGO, "d_obj": D_OBJ_SEQ, "d_ctx_v1": D_CTX_V1,
        "split": split,
        "test_results": results,
        "best_val_auroc": best.best_auroc, "best_val_auprc": best.best_auprc,
    }, ckpt_path)
    log.info(f"      Saved checkpoint to {ckpt_path}")

    with open(RESULTS_DIR / "summary_no_event.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[6/6] wrote {RESULTS_DIR / 'summary_no_event.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
