#!/usr/bin/env python3
"""v1 training variant: DeepAccident ONLY (no Bench2Drive), 4s window,
ego/object/road feature schema (features_v1.py).

Rationale (per explicit direction, not inferred): Bench2Drive has no native
collision ground truth at all -- only this package's own geometric SAT-
overlap detector, which is a weak, noisy training/eval signal on its own
(no independent cross-check exists for Bench2Drive the way DeepAccident's
native collision_intensity annotation lets the geometric detector be
verified/corrected). Concentrating training and evaluation on DeepAccident,
which does have trustworthy native collision labels, removes that
uncertainty from both stages rather than only from evaluation (as in the
v0 cross-domain protocol).

horizon_s=4.0 (vs. v0's 5.0) matches DeepAccident's own native record span
almost exactly (~4.1s: 1.0s past+current + 3.0s future, adapters/
deepaccident.py's docstring) -- v0's 5.0s window required padding beyond
what DeepAccident actually provides; this doesn't.

Does NOT touch any v0 artifact: separate checkpoint
(carm_deepaccident_v1.pt), separate results dir (results_v1/), separate
feature/dataset modules (features_v1.py/dataset_v1.py). Reuses the
existing DeepAccident-preprocessed cache as-is (tau_3d/interaction_type/
evasive_severity/tau_phy/T_evt are all per-frame quantities computed
independently of horizon_s/num_bins -- only feature-building and the
model's bin grid change here, not preprocessing).
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

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
LOGS_DIR = Path("dataset/logs")
RESULTS_DIR = Path("dataset/results_v1")

V1_CFG = FutureRiskConfig(horizon_s=4.0, num_bins=20)

STAGE1_EPOCHS = 40
STAGE2_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("train_deepaccident_v1")


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


def apply_native_target_override(datasets: dict, native_labels: dict) -> int:
    """Anchor the L_risk training target to DeepAccident's own native
    ground truth instead of only the geometric-derived analytic proxy.

    Per explicit direction: don't let the two signals blend/confuse each
    other. The rule is one-directional and simple, not a heuristic patch:
    - native says collision happened -> target forced to 1.0 (unambiguous;
      no geometry-based margin computation can override a true collision
      the geometric SAT-overlap detector simply never predicted -- this is
      exactly the failure mode found earlier: ~69% of DeepAccident's true
      collisions have no finite tau_3D at all, so `y_analytic` silently
      sat near 0 for genuinely-crashed instances).
    - native says no collision -> keep the continuous geometric-margin
      `y_analytic` as-is. Native is only a binary fact; it carries no
      information about *how close* a non-colliding trajectory came, which
      is exactly the continuous severity signal FutureRisk exists to
      capture (collision rate's blind spot) -- there is nothing to correct
      here, and forcing these instances to a binary target would throw
      away that continuous signal entirely.

    Mutates each dataset's cached items in place (`item["y_analytic"]` is
    already a scalar, not something feature-building would need to
    recompute) and returns the number of instances actually overridden.
    """
    n_overridden = 0
    for ds in datasets.values():
        for item in ds.items:
            entry = native_labels.get(item["instance_id"])
            if entry is not None and entry.get("collision_label") == 1 and item["y_analytic"] < 1.0:
                item["y_analytic"] = np.float32(1.0)
                n_overridden += 1
    return n_overridden


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
        loss = torch.nn.functional.binary_cross_entropy(R_instance.clamp(1e-6, 1 - 1e-6), batch.y_analytic)
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
def predict_split_detailed(model: CARM, loader, native_labels: dict) -> list:
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
            iid = batch.instance_ids[b]
            native = native_labels.get(iid)
            rows.append({
                "instance_id": iid,
                "y_analytic": float(batch.y_analytic[b].item()),
                "R_predicted": float(R_instance[b].item()),
                "collision_label_geometric": int(batch.collision_label[b].item()),
                "collision_label_native": int(native["collision_label"]) if native and native["collision_label"] is not None else None,
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


def main():
    set_seed(SEED)
    t0 = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_deepaccident_v1_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    log.info(f"v1 config: horizon_s={V1_CFG.horizon_s}, num_bins={V1_CFG.num_bins}"
             f" (bin width={V1_CFG.horizon_s/V1_CFG.num_bins:.2f}s), DeepAccident-only")

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
    log.info(f"      pos_weight(L_evt)={pos_weight:.2f}  risk_pos_weight(L_risk)={risk_pos_weight:.2f}")

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
    log.info("      Stage 1 (L_evt + L_risk + L_saturation):")
    for epoch in range(STAGE1_EPOCHS):
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt1.zero_grad()
            out = stage1_step(model, batch, cfg=V1_CFG)
            out["loss"].backward()
            opt1.step()
            epoch_losses.append(out["loss"].item())
        val_metrics = evaluate_split(model, val_loader, V1_CFG)
        is_best = best.maybe_update(model, val_metrics, stage=1, epoch=epoch)
        log.info(f"        stage1 epoch {epoch:3d}  train_loss={np.mean(epoch_losses):.4f}"
                 f"  val_L_risk={val_metrics['mean_L_risk']:.4f}"
                 f"  val_auroc={val_metrics['collision_auroc']:.3f}"
                 f"  val_auprc={val_metrics['collision_auprc']:.3f}"
                 f"  (n={val_metrics['n_instances']}, collisions={val_metrics['n_collisions']})"
                 f"{'  *best*' if is_best else ''}")

    model.set_stage(2)
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("      Stage 2 (L_risk only):")
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
    save_csv(test_rows, RESULTS_DIR / "predictions_test.csv")

    R = np.array([r["R_predicted"] for r in test_rows])
    y = np.array([r["y_analytic"] for r in test_rows])
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

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"[6/6] wrote {RESULTS_DIR / 'summary.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
