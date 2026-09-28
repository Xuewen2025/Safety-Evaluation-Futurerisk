#!/usr/bin/env python3
"""Post-hoc temperature scaling for the v1 (DeepAccident-only,
native-anchored) checkpoint: fit a single scalar T on the VAL split
(native label) minimizing NLL in logit space, apply to TEST predictions,
and report ECE before/after (AUROC/AUPRC are unaffected -- temperature
scaling is a monotonic transform of R, so it can't change ranking-based
metrics, only how well the score matches empirical frequency).

Motivated by a real, measured finding: v1's test-set reliability diagram
showed clear overconfidence (e.g. predicted confidence ~0.85-0.97 in the
top bins vs. empirical collision frequency only ~0.40-0.59 there) -- a
side effect of the native-anchoring fix forcing a hard 1.0 target whenever
native says a collision happened, which BCE rewards with extreme
confidence even though "a collision happened" doesn't mean every such
instance was equally certain/severe.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.evaluation import discrimination_metrics, reliability_diagram
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R, DEVICE

DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")


def setup_logging(log_path: Path):
    log = logging.getLogger("calibrate_v1_temperature")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """1D grid + local refinement search for T>0 minimizing BCE NLL --
    no autograd needed for a single scalar."""
    labels = labels.astype(float)

    def nll(T):
        p = np.clip(_sigmoid(logits / T), 1e-6, 1 - 1e-6)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    Ts = np.linspace(0.2, 5.0, 200)
    losses = [nll(T) for T in Ts]
    best_T = float(Ts[int(np.argmin(losses))])
    # local refine
    for _ in range(3):
        Ts2 = np.linspace(max(0.05, best_T - 0.1), best_T + 0.1, 41)
        losses2 = [nll(T) for T in Ts2]
        best_T = float(Ts2[int(np.argmin(losses2))])
    return best_T


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"calibrate_v1_temperature_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/3] Loading v1 checkpoint and scoring VAL split (native label) ...")
    model, cfg, ckpt = load_checkpoint()
    split = ckpt["split"]
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)

    def score_split(ids):
        R, lbl = [], []
        for iid in ids:
            p = Path(DEEPACCIDENT_PREPROCESSED) / f"{iid}.npz"
            if not p.exists():
                continue
            pre = load_preprocessed_instance(str(p))
            if pre.ego_box is None:
                continue
            entry = native_labels.get(iid)
            if entry is None or entry.get("collision_label") is None:
                continue
            raw = raw_instance_from_preprocessed(pre)
            R.append(predict_R(model, cfg, raw, pre))
            lbl.append(int(entry["collision_label"]))
        return np.array(R), np.array(lbl)

    R_val, lbl_val = score_split(split["val"])
    log.info(f"      val: n={len(R_val)} collisions={int(lbl_val.sum())}")

    log.info("[2/3] Fitting temperature T on VAL (native label) ...")
    logits_val = _logit(R_val)
    T = fit_temperature(logits_val, lbl_val)
    log.info(f"      fitted T={T:.4f}")

    rd_val_before = reliability_diagram(R_val, lbl_val, n_bins=10)
    R_val_calibrated = _sigmoid(logits_val / T)
    rd_val_after = reliability_diagram(R_val_calibrated, lbl_val, n_bins=10)
    log.info(f"      val ECE before={rd_val_before.ece:.4f}  after={rd_val_after.ece:.4f}")

    log.info("[3/3] Applying T to TEST predictions ...")
    R_test, lbl_test = score_split(split["test"])
    logits_test = _logit(R_test)
    R_test_calibrated = _sigmoid(logits_test / T)

    m_before = discrimination_metrics(R_test, lbl_test)
    m_after = discrimination_metrics(R_test_calibrated, lbl_test)
    rd_before = reliability_diagram(R_test, lbl_test, n_bins=10)
    rd_after = reliability_diagram(R_test_calibrated, lbl_test, n_bins=10)

    log.info(f"      TEST AUROC before={m_before['auroc']:.4f} after={m_after['auroc']:.4f}"
             f" (should match -- monotonic transform)")
    log.info(f"      TEST AUPRC before={m_before['auprc']:.4f} after={m_after['auprc']:.4f}")
    log.info(f"      TEST ECE   before={rd_before.ece:.4f} after={rd_after.ece:.4f}")
    for i in range(len(rd_after.bin_counts)):
        if rd_after.bin_counts[i] > 0:
            log.info(f"        bin [{rd_after.bin_edges[i]:.1f},{rd_after.bin_edges[i+1]:.1f})"
                     f" n={rd_after.bin_counts[i]:4d} conf={rd_after.bin_confidence[i]:.3f}"
                     f" emp_freq={rd_after.bin_accuracy[i]:.3f}")

    with open(RESULTS_DIR / "temperature_calibration.json", "w") as f:
        json.dump({
            "temperature": T,
            "val_ece_before": rd_val_before.ece, "val_ece_after": rd_val_after.ece,
            "test_auroc_before": m_before["auroc"], "test_auroc_after": m_after["auroc"],
            "test_auprc_before": m_before["auprc"], "test_auprc_after": m_after["auprc"],
            "test_ece_before": rd_before.ece, "test_ece_after": rd_after.ece,
        }, f, indent=2)
    log.info(f"      wrote {RESULTS_DIR / 'temperature_calibration.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
