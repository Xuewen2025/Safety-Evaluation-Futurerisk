#!/usr/bin/env python3
"""Reverse-direction cross-domain check for v1: the CARM checkpoint trained
ONLY on DeepAccident (native-anchored target, 4s window, ego/object/road
features -- `train_deepaccident_v1.py`) evaluated zero-shot on Bench2Drive,
which it has never seen. Mirrors v0's Bench2Drive->DeepAccident protocol
(evaluate_crossdomain_deepaccident.py) but in the opposite direction.

Ground truth here is Bench2Drive's own geometric collision label -- the
only one that exists for this dataset (no native annotation, unlike
DeepAccident); this is the same caveat already documented throughout this
project (README's Bench2Drive adapter section).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.carm_model import CARM, aggregate_instance_risk
from futurerisk.config import FutureRiskConfig
from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics, group_discrimination_metrics
from futurerisk.features_v1 import build_instance_batch_item_v1
from futurerisk.interaction import TYPE_NAMES

CHECKPOINT_PATH = "dataset/checkpoints/carm_deepaccident_v1.pt"
BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("evaluate_crossdomain_v1_to_bench2drive")


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


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = FutureRiskConfig(**{k: v for k, v in ckpt["cfg"].items() if k != "physical_prior_by_class"})
    model = CARM(
        d_kin=ckpt["d_ego"], d_int=ckpt["d_obj"], d_ctx=ckpt["d_ctx_v1"], cfg=cfg,
        hidden_dim=64, rnn_hidden=64, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def predict_R(model: CARM, cfg: FutureRiskConfig, raw, pre) -> float:
    item = build_instance_batch_item_v1(raw, pre, cfg)
    if item["x_ego"].shape[0] == 0:
        return 0.0
    x_ego = torch.from_numpy(item["x_ego"]).to(DEVICE)
    X_obj = torch.from_numpy(item["X_obj"]).to(DEVICE)
    x_ctx_v1 = torch.from_numpy(item["x_ctx_v1"]).to(DEVICE)
    tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
    out = model(x_kin=x_ego, X_int=X_obj, x_ctx=x_ctx_v1, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
    R_instance = aggregate_instance_risk(out["R"].unsqueeze(0))
    return float(R_instance.item())


def instance_dominant_type(pre) -> str:
    if not pre.tau_3d:
        return TYPE_NAMES[4]
    worst_agent = min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))
    code = pre.dominant_interaction_type.get(worst_agent, 4)
    return TYPE_NAMES.get(code, "other")


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"evaluate_crossdomain_v1_to_bench2drive_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    log.info(f"[1/3] Loading checkpoint {CHECKPOINT_PATH} (DeepAccident-only, v1) ...")
    model, cfg, ckpt = load_checkpoint()
    log.info(f"      horizon_s={cfg.horizon_s} num_bins={cfg.num_bins}"
             f" best_val_auroc={ckpt.get('best_val_auroc')} best_val_auprc={ckpt.get('best_val_auprc')}")

    log.info(f"[2/3] Scoring every Bench2Drive-Base instance (zero-shot) from {BENCH2DRIVE_PREPROCESSED} ...")
    rows = []
    for i, p in enumerate(sorted(Path(BENCH2DRIVE_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        R_pred = predict_R(model, cfg, raw, pre)
        geometric_collision_label = int(any(f.any() for f in raw.agent_collision_flags.values()))
        rows.append({
            "instance_id": pre.instance_id, "R_predicted": R_pred,
            "geometric_collision_label": geometric_collision_label,
            "dominant_type": instance_dominant_type(pre),
        })
        if (i + 1) % 1000 == 0:
            log.info(f"      ... {i+1} instances scored")
    log.info(f"      -> {len(rows)} Bench2Drive instances scored")

    R = np.array([r["R_predicted"] for r in rows])
    lbl = np.array([r["geometric_collision_label"] for r in rows])
    m = discrimination_metrics(R, lbl)
    log.info(f"[3/3] Zero-shot DeepAccident(v1) -> Bench2Drive: n={len(rows)} collisions={int(lbl.sum())}"
             f" ({100*lbl.mean():.1f}%)  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")

    groups = np.array([r["dominant_type"] for r in rows])
    per_type = group_discrimination_metrics(R, lbl, groups)
    log.info("      per-interaction-type:")
    for t, v in per_type.items():
        log.info(f"        {t:24s} n={int(v['n']):5d}  AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}")

    summary = {"n": len(rows), "n_collisions": int(lbl.sum()), "auroc": m["auroc"], "auprc": m["auprc"],
               "per_interaction_type": per_type}
    with open(RESULTS_DIR / "crossdomain_v1_to_bench2drive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"      wrote {RESULTS_DIR / 'crossdomain_v1_to_bench2drive_summary.json'}")
    log.info(f"Total wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
