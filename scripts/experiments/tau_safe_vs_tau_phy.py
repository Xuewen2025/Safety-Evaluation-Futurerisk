#!/usr/bin/env python3
"""experiment_design.txt 五.2: compare CARM's learned safe-TTC against the
analytic physical prior tau_phy, broken out by interaction type and
relative closing speed -- checks whether the learned correction moves in
directions consistent with domain intuition (e.g. oncoming conflicts
demanding a larger safety margin than the analytic prior alone provides).

Also serves as a live sanity check of 五.3's "interpretability by
construction" claim: mu_s = log(tau_phy) + delta_max*tanh(g_mu(h)), so the
learned median safe-TTC T_safe = exp(mu_s) = tau_phy * exp(correction) can
never leave [tau_phy/2, tau_phy*2] when delta_max=log(2) (this project's
default) -- verified numerically below, not just asserted.

No retraining needed: this only runs inference with the existing
(corrected-tau_phy) Bench2Drive-Base-only checkpoint over both datasets'
test/held-out instances.
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
from futurerisk.features import build_instance_batch_item
from futurerisk.interaction import TYPE_NAMES
from scripts.experiments.consistency_finegrained_eval import DEVICE, load_checkpoint

BENCH2DRIVE_PREPROCESSED = "dataset/Bench2Drive-Base-preprocessed"
DEEPACCIDENT_PREPROCESSED = "dataset/DeepAccident-preprocessed"
NATIVE_LABELS_PATH = "dataset/DeepAccident-preprocessed/native_collision_labels.json"
RESULTS_DIR = Path("dataset/results_experiments")
LOGS_DIR = Path("dataset/logs")

log = logging.getLogger("tau_safe_vs_tau_phy")


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


@torch.no_grad()
def per_agent_tau_comparison(model, cfg, raw, pre) -> list:
    """One row per agent in this instance: interaction type, mean
    v_rel_close (proxy for relative closing speed), tau_phy and the
    learned T_safe=exp(mu_s), both averaged over the bin grid to one
    representative scalar per agent-instance."""
    item = build_instance_batch_item(raw, pre, cfg)
    if item["x_kin"].shape[0] == 0:
        return []
    x_kin = torch.from_numpy(item["x_kin"]).to(DEVICE)
    X_int = torch.from_numpy(item["X_int"]).to(DEVICE)
    x_ctx = torch.from_numpy(item["x_ctx"]).to(DEVICE)
    tau_phy_seq = torch.from_numpy(item["tau_phy_seq"]).to(DEVICE)
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32, device=DEVICE)
    out = model(x_kin=x_kin, X_int=X_int, x_ctx=x_ctx, tau_phy_seq=tau_phy_seq, bin_edges=bin_edges)
    mu_s = out["mu_s"].cpu().numpy()          # (J, K)
    tau_phy_np = item["tau_phy_seq"]          # (J, K)
    t_safe_learned = np.exp(mu_s)             # (J, K), real-space median safe-TTC

    from futurerisk.data_pipeline import _relative_closing_speed
    rows = []
    for j, aid in enumerate(item["agent_ids"]):
        agent_box = raw.agent_boxes[aid]
        v_close = _relative_closing_speed(raw.ego_box, agent_box, cfg.dt)
        itype_code = pre.dominant_interaction_type.get(aid, 4)
        rows.append({
            "instance_id": pre.instance_id, "agent_id": aid, "dataset": pre.dataset,
            "interaction_type": TYPE_NAMES.get(itype_code, "other"),
            "mean_v_rel_close": float(np.mean(v_close)),
            "mean_tau_phy": float(np.mean(tau_phy_np[j])),
            "mean_T_safe_learned": float(np.mean(t_safe_learned[j])),
        })
    return rows


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"tau_safe_vs_tau_phy_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_path)
    log.info(f"Log file: {log_path}")
    t0 = time.time()

    model, cfg, ckpt = load_checkpoint()
    log.info(f"delta_max={cfg.delta_max:.4f} (learned T_safe bounded to "
             f"[tau_phy*{np.exp(-cfg.delta_max):.3f}, tau_phy*{np.exp(cfg.delta_max):.3f}])")

    all_rows = []
    log.info("[1/2] Bench2Drive test split ...")
    for iid in ckpt["split"]["test"]:
        p = Path(BENCH2DRIVE_PREPROCESSED) / f"{iid}.npz"
        if not p.exists():
            continue
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None:
            continue
        raw = raw_instance_from_preprocessed(pre)
        all_rows.extend(per_agent_tau_comparison(model, cfg, raw, pre))

    log.info("[2/2] DeepAccident (all instances with native label) ...")
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    for i, p in enumerate(sorted(Path(DEEPACCIDENT_PREPROCESSED).glob("*.npz"))):
        pre = load_preprocessed_instance(str(p))
        if pre.ego_box is None or pre.instance_id not in native_labels:
            continue
        raw = raw_instance_from_preprocessed(pre)
        all_rows.extend(per_agent_tau_comparison(model, cfg, raw, pre))
        if (i + 1) % 2000 == 0:
            log.info(f"      ... {i+1} files scanned, {len(all_rows)} agent-rows so far")

    log.info(f"-> {len(all_rows)} agent-instance rows total")

    # Verify the bounded-by-construction claim numerically (五.3), not just assert it.
    ratios = np.array([r["mean_T_safe_learned"] / max(r["mean_tau_phy"], 1e-6) for r in all_rows])
    lo, hi = np.exp(-cfg.delta_max), np.exp(cfg.delta_max)
    n_out_of_bound = int(((ratios < lo - 1e-4) | (ratios > hi + 1e-4)).sum())
    log.info(f"Bounded-residual check: T_safe/tau_phy ratio range observed=[{ratios.min():.3f}, {ratios.max():.3f}],"
             f" allowed=[{lo:.3f}, {hi:.3f}], out-of-bound rows={n_out_of_bound}/{len(ratios)}")

    # Bin by (interaction_type, relative-speed tercile) -- report mean
    # offset direction (log-ratio, sign-agnostic to which side dominates).
    v_close_vals = np.array([r["mean_v_rel_close"] for r in all_rows])
    terciles = np.quantile(v_close_vals, [1 / 3, 2 / 3])

    def speed_bin(v):
        if v <= terciles[0]:
            return "receding/slow"
        if v <= terciles[1]:
            return "medium"
        return "fast-closing"

    from collections import defaultdict
    groups = defaultdict(list)
    for r, ratio in zip(all_rows, ratios):
        key = (r["interaction_type"], speed_bin(r["mean_v_rel_close"]))
        groups[key].append((r["mean_tau_phy"], r["mean_T_safe_learned"], ratio))

    log.info("Per (interaction_type, relative-speed bin): mean tau_phy vs. mean learned T_safe, log2-ratio:")
    summary_rows = []
    for key in sorted(groups.keys()):
        vals = groups[key]
        n = len(vals)
        mean_phy = float(np.mean([v[0] for v in vals]))
        mean_learned = float(np.mean([v[1] for v in vals]))
        mean_log2_ratio = float(np.mean(np.log2([v[2] for v in vals])))
        log.info(f"    {key[0]:24s} {key[1]:16s} n={n:6d}  tau_phy={mean_phy:.3f}s"
                 f"  T_safe_learned={mean_learned:.3f}s  log2(ratio)={mean_log2_ratio:+.3f}"
                 f" ({'more conservative' if mean_log2_ratio > 0 else 'less conservative'} than prior)")
        summary_rows.append({
            "interaction_type": key[0], "speed_bin": key[1], "n": n,
            "mean_tau_phy": mean_phy, "mean_T_safe_learned": mean_learned, "mean_log2_ratio": mean_log2_ratio,
        })

    with open(RESULTS_DIR / "tau_safe_vs_tau_phy_summary.json", "w") as f:
        json.dump({
            "delta_max": cfg.delta_max, "bound_lo": lo, "bound_hi": hi,
            "n_out_of_bound": n_out_of_bound, "n_total": len(ratios),
            "groups": summary_rows,
        }, f, indent=2)
    log.info(f"wrote {RESULTS_DIR / 'tau_safe_vs_tau_phy_summary.json'}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
