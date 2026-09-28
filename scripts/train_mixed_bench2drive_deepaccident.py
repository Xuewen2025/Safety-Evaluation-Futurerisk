#!/usr/bin/env python3
"""Mixed training: Bench2Drive-Base + DeepAccident combined, each split
independently at the clip level into train/val/test, then pooled --
the "step 4" comparison against the Bench2Drive-only model (this script's
predecessor) and the Bench2Drive-only-trained zero-shot cross-domain result
(evaluate_crossdomain_deepaccident.py).

Reports three test-set numbers separately (not just one pooled number):
    - pooled (both datasets' test instances together)
    - Bench2Drive-only test subset
    - DeepAccident-only test subset (using its NATIVE collision_intensity-
      derived label, not the geometric one -- see
      evaluate_crossdomain_deepaccident.py's tautology finding)
so "did mixing help, and did it cost anything on either individual
domain" can be answered directly rather than inferred from one blended
number.
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
from futurerisk.config import DEFAULT_CONFIG
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.dataset import CARMDataset, compute_pos_weight, compute_risk_pos_weight, make_collate_fn
from futurerisk.evaluation import discrimination_metrics
from futurerisk.features import D_CTX, D_INT, D_KIN
from futurerisk.split import split_instances_by_clip
from futurerisk.training import stage1_step, stage2_step

BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
CHECKPOINT_PATH = "dataset/checkpoints/carm_mixed_bench2drive_deepaccident.pt"
LOGS_DIR = Path("dataset/logs")
RESULTS_DIR = Path("dataset/results_mixed")

STAGE1_EPOCHS = 40
STAGE2_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("train_mixed")


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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BestCheckpointTracker:
    def __init__(self):
        self.best_auroc = -1.0
        self.best_auprc = -1.0
        self.state_dict = None
        self.stage = None
        self.epoch = None

    def maybe_update(self, model: CARM, val_metrics: dict, stage: int, epoch: int) -> bool:
        auroc, auprc = val_metrics["collision_auroc"], val_metrics["collision_auprc"]
        if np.isnan(auroc):
            return False
        better = (auroc > self.best_auroc) or (auroc == self.best_auroc and auprc > self.best_auprc)
        if better:
            self.best_auroc, self.best_auprc = auroc, auprc
            self.state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.stage, self.epoch = stage, epoch
        return better


def load_all_preprocessed(directory: str) -> dict:
    out = {}
    for p in Path(directory).glob("*.npz"):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is not None:
            out[pre.instance_id] = pre
    return out


@torch.no_grad()
def evaluate_split(model: CARM, loader, cfg) -> dict:
    model.eval()
    all_R, all_labels, losses = [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        B, J = batch.agent_mask.shape
        flat = lambda t: t.reshape(B * J, *t.shape[2:])
        out = model(
            x_kin=flat(batch.x_kin), X_int=flat(batch.X_int), x_ctx=flat(batch.x_ctx),
            tau_phy_seq=flat(batch.tau_phy_seq), bin_edges=batch.bin_edges,
        )
        R_per_agent = out["R"].view(B, J)
        R_instance = aggregate_instance_risk(R_per_agent, agent_mask=batch.agent_mask)
        loss = torch.nn.functional.binary_cross_entropy(
            R_instance.clamp(1e-6, 1 - 1e-6), batch.y_analytic,
        )
        losses.append(loss.item())
        all_R.extend(R_instance.tolist())
        all_labels.extend(batch.collision_label.tolist())
    model.train()
    metrics = {"mean_L_risk": float(np.mean(losses)) if losses else float("nan")}
    metrics.update({f"collision_{k}": v for k, v in discrimination_metrics(all_R, all_labels).items()})
    metrics["n_instances"] = len(all_R)
    metrics["n_collisions"] = int(sum(all_labels))
    return metrics


@torch.no_grad()
def predict_split_detailed(model: CARM, loader, cfg) -> list:
    model.eval()
    rows = []
    for batch in loader:
        batch = batch.to(DEVICE)
        B, J = batch.agent_mask.shape
        flat = lambda t: t.reshape(B * J, *t.shape[2:])
        out = model(
            x_kin=flat(batch.x_kin), X_int=flat(batch.X_int), x_ctx=flat(batch.x_ctx),
            tau_phy_seq=flat(batch.tau_phy_seq), bin_edges=batch.bin_edges,
        )
        R_per_agent = out["R"].view(B, J)
        R_instance = aggregate_instance_risk(R_per_agent, agent_mask=batch.agent_mask)
        for b in range(B):
            rows.append({
                "instance_id": batch.instance_ids[b],
                "dataset": "DeepAccident" if "__r" in batch.instance_ids[b] else "Bench2Drive",
                "num_agents": int(batch.agent_mask[b].sum().item()),
                "y_analytic": float(batch.y_analytic[b].item()),
                "R_predicted": float(R_instance[b].item()),
                "collision_label_geometric": int(batch.collision_label[b].item()),
            })
    model.train()
    return rows


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report_discrimination(name: str, rows: list, native_labels: dict) -> dict:
    """For Bench2Drive rows, use the geometric label (no native annotation
    exists for that dataset). For DeepAccident rows, use the native label
    (falls back to geometric only if somehow missing)."""
    R, lbl = [], []
    for r in rows:
        R.append(r["R_predicted"])
        if r["dataset"] == "DeepAccident":
            entry = native_labels.get(r["instance_id"])
            lbl.append(int(entry["collision_label"]) if entry is not None else r["collision_label_geometric"])
        else:
            lbl.append(r["collision_label_geometric"])
    R, lbl = np.array(R), np.array(lbl)
    m = discrimination_metrics(R, lbl)
    log.info(f"      {name:30s}  n={len(rows):5d}  collisions={int(lbl.sum()):5d} ({100*lbl.mean():.1f}%)"
             f"  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}  mean_R|c=1={R[lbl==1].mean() if lbl.sum() else float('nan'):.3f}"
             f"  mean_R|c=0={R[lbl==0].mean() if (lbl==0).any() else float('nan'):.3f}")
    return {"n": len(rows), "n_collisions": int(lbl.sum()), "auroc": m["auroc"], "auprc": m["auprc"]}


def main():
    set_seed(SEED)
    t0 = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_mixed_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    log.info("[1/6] Loading cached preprocessing for both datasets ...")
    bd_pre = load_all_preprocessed(BENCH2DRIVE_PREPROCESSED)
    da_pre = load_all_preprocessed(DEEPACCIDENT_PREPROCESSED)
    log.info(f"      Bench2Drive: {len(bd_pre)} instances   DeepAccident: {len(da_pre)} instances")

    bd_split = split_instances_by_clip(list(bd_pre.keys()), train_frac=0.6, val_frac=0.2, seed=SEED)
    da_split = split_instances_by_clip(list(da_pre.keys()), train_frac=0.6, val_frac=0.2, seed=SEED)
    log.info(f"[2/6] Per-dataset clip-level splits:")
    for name, s in (("Bench2Drive", bd_split), ("DeepAccident", da_split)):
        log.info(f"      {name}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")

    pre_by_id = {**bd_pre, **da_pre}
    raw_by_id = {iid: raw_instance_from_preprocessed(pre) for iid, pre in pre_by_id.items()}

    mixed_split = {
        part: bd_split[part] + da_split[part] for part in ("train", "val", "test")
    }
    log.info(f"      mixed: train={len(mixed_split['train'])} val={len(mixed_split['val'])}"
             f" test={len(mixed_split['test'])}")

    datasets = {
        part: CARMDataset([raw_by_id[i] for i in ids], pre_by_id, cfg=DEFAULT_CONFIG)
        for part, ids in mixed_split.items()
    }
    for part, ds in datasets.items():
        log.info(f"      {part} dataset (>=1 agent): {len(ds)} instances")

    n_train_positive = sum(1 for item in datasets["train"].items if item["y_analytic"] > 0.5)
    log.info(f"      train instances with y_analytic > 0.5: {n_train_positive} / {len(datasets['train'])}")

    pos_weight = compute_pos_weight(datasets["train"])
    risk_pos_weight = compute_risk_pos_weight(datasets["train"])
    log.info(f"      pos_weight(L_evt)={pos_weight:.2f}  risk_pos_weight(L_risk)={risk_pos_weight:.2f}")

    collate = make_collate_fn(cfg=DEFAULT_CONFIG, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False, collate_fn=collate)

    log.info(f"[3/6] Building CARM (d_kin={D_KIN}, d_int={D_INT}, d_ctx={D_CTX}, dropout={DROPOUT})"
             f" on device={DEVICE}, training (weight_decay={WEIGHT_DECAY}) ...")
    model = CARM(
        d_kin=D_KIN, d_int=D_INT, d_ctx=D_CTX, cfg=DEFAULT_CONFIG,
        hidden_dim=64, rnn_hidden=64, dropout=DROPOUT,
    ).to(DEVICE)
    log.info(f"      model params: {sum(p.numel() for p in model.parameters())}")

    best = BestCheckpointTracker()

    model.set_stage(1)
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 1 (L_evt + L_risk):")
    for epoch in range(STAGE1_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt1.zero_grad()
            out = stage1_step(model, batch, cfg=DEFAULT_CONFIG)
            out["loss"].backward()
            opt1.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, DEFAULT_CONFIG)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        stage1 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_L_risk={val_metrics['mean_L_risk']:.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"  (n={val_metrics['n_instances']}, collisions={val_metrics['n_collisions']})"
                 f"{'  *best*' if is_best else ''}")

    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 2 (L_risk only; L_rank skipped, no matched-scenario-pair bookkeeping):")
    for epoch in range(STAGE2_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt2.zero_grad()
            out = stage2_step(model, batch, cfg=DEFAULT_CONFIG)
            out["loss"].backward()
            opt2.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, DEFAULT_CONFIG)
        is_best = best.maybe_update(model, val_metrics, stage=2, epoch=epoch)
        log.info(f"        stage2 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_L_risk={val_metrics['mean_L_risk']:.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"{'  *best*' if is_best else ''}")

    log.info(f"[4/6] Best checkpoint: stage={best.stage} epoch={best.epoch}"
             f" val_auroc={best.best_auroc:.3f} val_auprc={best.best_auprc:.3f} -- restoring it")
    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})

    log.info("[5/6] Test-set evaluation: pooled, Bench2Drive-only, DeepAccident-only (native label) ...")
    test_rows = predict_split_detailed(model, test_loader, DEFAULT_CONFIG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(test_rows, RESULTS_DIR / "predictions_test_pooled.csv")

    bd_test_rows = [r for r in test_rows if r["dataset"] == "Bench2Drive"]
    da_test_rows = [r for r in test_rows if r["dataset"] == "DeepAccident"]
    results = {}
    results["pooled"] = report_discrimination("pooled (both datasets)", test_rows, native_labels)
    results["bench2drive_only"] = report_discrimination("Bench2Drive-only test", bd_test_rows, native_labels)
    results["deepaccident_only"] = report_discrimination("DeepAccident-only test (native)", da_test_rows, native_labels)

    ckpt_path = Path(CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "cfg": dataclasses.asdict(DEFAULT_CONFIG),
        "d_kin": D_KIN, "d_int": D_INT, "d_ctx": D_CTX,
        "bd_split": bd_split, "da_split": da_split,
        "test_results": results,
        "best_val_auroc": best.best_auroc, "best_val_auprc": best.best_auprc,
    }, ckpt_path)
    log.info(f"      Saved checkpoint to {ckpt_path}")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[6/6] wrote {RESULTS_DIR / 'summary.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
