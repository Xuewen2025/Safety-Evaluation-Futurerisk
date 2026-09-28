#!/usr/bin/env python3
"""End-to-end CARM training run on Bench2Drive-Base (~1000 clips, ~3679
five-second windows) -- the full-scale successor to
`train_bench2drive.py`'s Mini smoke test.

Unlike the Mini script, this one never touches the original .tar.gz
archives: `data_pipeline.PreprocessedInstance` now caches the raw
trajectories too (see data_pipeline.py's `raw_instance_from_preprocessed`),
so everything needed for feature-tensor assembly comes straight out of
`dataset/Bench2Drive-Base-preprocessed/*.npz`.

Produces the same three kinds of durable output as the Mini run:
    dataset/logs/train_bench2drive_base_<timestamp>.log
    dataset/results_base/predictions_{train,val,test}.csv
    dataset/results_base/case_study.png
    dataset/checkpoints/carm_bench2drive_base.pt
"""
from __future__ import annotations

import csv
import dataclasses
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
from futurerisk.features import D_CTX, D_INT, D_KIN, build_instance_batch_item
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk
from futurerisk.split import split_instances_by_clip
from futurerisk.training import stage1_step, stage2_step
from futurerisk.visualization import case_study_grid

PREPROCESSED_DIR = "dataset/Bench2Drive-Base-preprocessed"
CHECKPOINT_PATH = "dataset/checkpoints/carm_bench2drive_base.pt"
LOGS_DIR = Path("dataset/logs")
RESULTS_DIR = Path("dataset/results_base")

STAGE1_EPOCHS = 40
STAGE2_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("train_bench2drive_base")


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


class BestCheckpointTracker:
    """Tracks the best-validation-AUROC (tie-break AUPRC) model snapshot
    seen across *both* training stages, so the deliverable model is the
    best one actually observed during training rather than whatever the
    fixed epoch budget happens to end on. Does not interrupt training or
    reload mid-run -- Stage 2 still starts from Stage 1's true final
    weights, as S3.8 specifies; this only decides what gets evaluated/
    saved/visualized at the very end."""

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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                "num_agents": int(batch.agent_mask[b].sum().item()),
                "y_analytic": float(batch.y_analytic[b].item()),
                "R_predicted": float(R_instance[b].item()),
                "collision_label": int(batch.collision_label[b].item()),
            })
    model.train()
    return rows


def save_predictions_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def build_case_study_entry(instance_id, raw_by_id, pre_by_id, model, cfg) -> dict:
    raw, pre = raw_by_id[instance_id], pre_by_id[instance_id]
    margins = [context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d]
    margin_t = np.min(np.stack(margins), axis=0) if margins else np.full(len(pre.times), np.inf)
    risk_t = instantaneous_risk(margin_t, T_m=cfg.T_m)

    item = build_instance_batch_item(raw, pre, cfg)
    if item["x_kin"].shape[0] > 0:
        x_kin = torch.from_numpy(item["x_kin"]).to(DEVICE)
        X_int = torch.from_numpy(item["X_int"]).to(DEVICE)
        x_ctx = torch.from_numpy(item["x_ctx"]).to(DEVICE)
        tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
        bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
        model.eval()
        out = model(x_kin=x_kin, X_int=X_int, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
        model.train()
        q = out["q"].cpu().numpy()
        q_combined = 1.0 - np.prod(1.0 - q, axis=0)
    else:
        q_combined = np.zeros(cfg.num_bins)

    bin_centers = 0.5 * (cfg.bin_edges[:-1] + cfg.bin_edges[1:])
    collision_free = not any(flags.any() for flags in raw.agent_collision_flags.values())
    return dict(
        times=pre.times, margin_t=margin_t, risk_t=risk_t,
        bin_centers=bin_centers, q_k=q_combined,
        title=instance_id, collision_free=collision_free,
    )


def select_case_study_instances(raw_by_id: dict, pre_by_id: dict, max_examples: int = 3) -> list:
    worst_margin = {}
    collision_free_ids, collided_ids = [], []
    for iid, pre in pre_by_id.items():
        if iid not in raw_by_id:
            continue
        margins = [context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d]
        if not margins:
            continue
        worst_margin[iid] = float(np.min(np.concatenate(margins)))
        raw = raw_by_id[iid]
        is_collision = any(flags.any() for flags in raw.agent_collision_flags.values())
        (collided_ids if is_collision else collision_free_ids).append(iid)

    chosen = []
    if collision_free_ids:
        by_margin = sorted(collision_free_ids, key=lambda i: worst_margin[i])
        chosen.append(by_margin[0])
        if by_margin[-1] != by_margin[0]:
            chosen.append(by_margin[-1])
    if collided_ids:
        chosen.append(sorted(collided_ids, key=lambda i: worst_margin[i])[0])
    return chosen[:max_examples]


def main():
    set_seed(SEED)
    t0 = time.time()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_bench2drive_base_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")

    log.info(f"[1/6] Loading cached preprocessing (raw + labels) from {PREPROCESSED_DIR} ...")
    pre_by_id, raw_by_id = {}, {}
    n_skipped_no_raw = 0
    for p in Path(PREPROCESSED_DIR).glob("*.npz"):
        pre = load_preprocessed_instance(str(p))
        pre_by_id[pre.instance_id] = pre
        if pre.ego_box is None:
            n_skipped_no_raw += 1
            continue
        raw_by_id[pre.instance_id] = raw_instance_from_preprocessed(pre)
    log.info(f"      -> {len(pre_by_id)} cached instances loaded, {len(raw_by_id)} with usable raw trajectories"
             f" ({n_skipped_no_raw} missing raw fields, skipped)")

    all_ids = [iid for iid in raw_by_id if iid in pre_by_id]
    split = split_instances_by_clip(all_ids, train_frac=0.6, val_frac=0.2, seed=SEED)
    log.info(f"[2/6] Clip-level split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} instances")
    for name, ids in split.items():
        n_clips = len(set(i.split('__w')[0] for i in ids))
        log.info(f"      {name}: {len(ids)} instances from {n_clips} clips")

    datasets = {
        name: CARMDataset([raw_by_id[i] for i in ids], pre_by_id, cfg=DEFAULT_CONFIG)
        for name, ids in split.items()
    }
    for name, ds in datasets.items():
        log.info(f"      {name} dataset: {len(ds)} instances with >=1 agent")

    n_train_positive = sum(1 for item in datasets["train"].items if item["y_analytic"] > 0.5)
    log.info(f"      train instances with y_analytic > 0.5: {n_train_positive} / {len(datasets['train'])}")

    pos_weight = compute_pos_weight(datasets["train"])
    risk_pos_weight = compute_risk_pos_weight(datasets["train"])
    log.info(f"      pos_weight (L_evt, inverse-frequency, train split) = {pos_weight:.2f}")
    log.info(f"      risk_pos_weight (L_risk, inverse-frequency, train split) = {risk_pos_weight:.2f}")

    collate = make_collate_fn(cfg=DEFAULT_CONFIG, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False, collate_fn=collate)

    log.info(f"[3/6] Building CARM (d_kin={D_KIN}, d_int={D_INT}, d_ctx={D_CTX}, dropout={DROPOUT})"
             f" on device={DEVICE} and training (weight_decay={WEIGHT_DECAY}) ...")
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
    log.info("      Stage 2 (L_risk only; L_rank skipped, no matched-scenario-pair bookkeeping yet):")
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
             f" val_auroc={best.best_auroc:.3f} val_auprc={best.best_auprc:.3f}"
             f" -- restoring it before final eval (final-epoch weights are discarded if not also the best)")
    if best.state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best.state_dict.items()})
    else:
        log.info("      no valid best checkpoint found (all-NaN val AUROC?) -- keeping final-epoch weights")

    log.info("Final evaluation, per-instance predictions, checkpoint:")
    test_metrics = evaluate_split(model, test_loader, DEFAULT_CONFIG)
    log.info(f"      test metrics: {test_metrics}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        rows = predict_split_detailed(model, loader, DEFAULT_CONFIG)
        out_path = RESULTS_DIR / f"predictions_{name}.csv"
        save_predictions_csv(rows, out_path)
        log.info(f"      wrote {len(rows)} rows -> {out_path}")

    ckpt_path = Path(CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": cpu_state_dict,
        "cfg": dataclasses.asdict(DEFAULT_CONFIG),
        "d_kin": D_KIN, "d_int": D_INT, "d_ctx": D_CTX,
        "split": split,
        "test_metrics": test_metrics,
        "best_checkpoint_stage": best.stage,
        "best_checkpoint_epoch": best.epoch,
        "best_val_auroc": best.best_auroc,
        "best_val_auprc": best.best_auprc,
    }, ckpt_path)
    log.info(f"      Saved checkpoint to {ckpt_path} (this IS the best-val snapshot, not necessarily the last epoch)")

    log.info("[5/6] Case-study visualization (S3.11) ...")
    case_ids = select_case_study_instances(raw_by_id, pre_by_id)
    log.info(f"      selected instances: {case_ids}")
    case_entries = [build_case_study_entry(iid, raw_by_id, pre_by_id, model, DEFAULT_CONFIG) for iid in case_ids]
    plot_path = RESULTS_DIR / "case_study.png"
    case_study_grid(case_entries, save_path=str(plot_path))
    log.info(f"      wrote {plot_path}")

    log.info(f"[6/6] Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
