#!/usr/bin/env python3
"""Bird's-eye-view trajectory visualization: pick instances spanning the
model's predicted-risk score distribution (from the already-saved test-set
predictions), plot each one's actual ego + agent paths in space (colored by
analytic risk r(t) along the path), and annotate each panel with the
model's score, the analytic target, and the true collision label -- so
scores can be sanity-checked against the geometry they were assigned to,
not just read off as numbers.

Reads (no retraining, no re-parsing archives):
    dataset/results_base/predictions_test.csv
    dataset/Bench2Drive-Base-preprocessed/*.npz   (cached raw trajectories)

Writes:
    dataset/results_base/trajectory_bev.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from futurerisk.data_pipeline import load_preprocessed_instance, raw_instance_from_preprocessed
from futurerisk.safety_prior import context_dependent_margin, instantaneous_risk
from futurerisk.visualization import bev_grid

PREDICTIONS_CSV = "dataset/results_base/predictions_test.csv"
PREPROCESSED_DIR = "dataset/Bench2Drive-Base-preprocessed"
OUT_PATH = "dataset/results_base/trajectory_bev.png"
N_PANELS = 9
MAX_AGENTS_PER_PANEL = 4


def load_predictions(path: str) -> list:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["R_predicted"] = float(r["R_predicted"])
        r["y_analytic"] = float(r["y_analytic"])
        r["collision_label"] = int(r["collision_label"])
        r["num_agents"] = int(r["num_agents"])
    return rows


def select_spread(rows: list, n: int) -> list:
    """Evenly spaced picks across the R_predicted-sorted distribution
    (0th, 1/(n-1), 2/(n-1), ..., 100th percentile by rank) -- a spread from
    lowest-scored to highest-scored, not just the extremes or a random
    sample, so the grid reads as a genuine low-to-high comparison."""
    rows_sorted = sorted(rows, key=lambda r: r["R_predicted"])
    if len(rows_sorted) <= n:
        return rows_sorted
    idx = np.linspace(0, len(rows_sorted) - 1, n).round().astype(int)
    seen, picked = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            picked.append(rows_sorted[i])
    return picked


def build_bev_entry(row: dict, pre) -> dict:
    ego = pre.ego_box
    ego_xy = np.stack([ego.x, ego.y], axis=-1)

    margins_by_agent = {aid: context_dependent_margin(pre.tau_3d[aid], pre.tau_phy[aid]) for aid in pre.tau_3d}
    margin_t = np.min(np.stack(list(margins_by_agent.values())), axis=0) if margins_by_agent else np.full(len(pre.times), np.inf)
    risk_t = instantaneous_risk(margin_t, T_m=1.0)

    # show the agents that actually got close (smallest own worst-case
    # margin first), not an arbitrary subset
    order = sorted(margins_by_agent, key=lambda aid: np.min(margins_by_agent[aid]))
    agent_xys = {}
    for aid in order[:MAX_AGENTS_PER_PANEL]:
        box = pre.agent_boxes[aid]
        agent_xys[aid] = np.stack([box.x, box.y], axis=-1)

    return dict(
        ego_xy=ego_xy, agent_xys=agent_xys, risk_t=risk_t,
        R_predicted=row["R_predicted"], y_analytic=row["y_analytic"],
        collision_label=row["collision_label"], title=row["instance_id"],
        max_agents=MAX_AGENTS_PER_PANEL,
    )


def main():
    rows = load_predictions(PREDICTIONS_CSV)
    print(f"Loaded {len(rows)} test-set predictions from {PREDICTIONS_CSV}")
    picked = select_spread(rows, N_PANELS)
    print(f"Selected {len(picked)} instances spanning the R_predicted distribution:")
    for r in picked:
        print(f"  R_pred={r['R_predicted']:.3f}  y_analytic={r['y_analytic']:.3f}"
              f"  collision={r['collision_label']}  {r['instance_id']}")

    entries = []
    for row in picked:
        p = Path(PREPROCESSED_DIR) / f"{row['instance_id']}.npz"
        pre = load_preprocessed_instance(str(p))
        entries.append(build_bev_entry(row, pre))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    bev_grid(entries, n_cols=3, save_path=OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
