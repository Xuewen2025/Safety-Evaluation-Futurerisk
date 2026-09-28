#!/usr/bin/env python3
"""Render root.tex fig:planner_slope_chart: a slopegraph replacing the old
tab:planner_level_ranking table, to make the UniAD/VAD rank crossover in
Sec. IV-C visually immediate rather than something a reader has to spot
from parenthetical rank numbers in a table.

Two independently-scaled vertical axes (left: mean R over each planner's
substituted trajectories; right: geometric collision rate over the same
trajectories), each planner's real value plotted on both and joined by a
straight line. Both axes run low-at-top (rank 1 = safest, matching the
table this replaces), so a line that stays roughly horizontal means the
two metrics agree on that planner's relative safety, and a crossing line
means they disagree. TCP and AD-MLP's lines run parallel (both metrics
agree on their order); UniAD's and VAD's lines cross -- the same swap
Sec. IV-C's prose already discusses via the AccidentTwoWays_Route1105
case study, now visible directly as a crossing rather than only as two
differing rank numbers in a table.

All four numbers are recomputed here from the same cached counterfactual
JSONs Table III (and Sec. IV-C's prose) were built from, not hand-copied,
so this figure cannot silently drift out of sync with the table it
replaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})

# (label, counterfactual-JSON path, R field, collision field)
SOURCES = [
    ("TCP", "dataset/results_v1/tcp_counterfactual_case_studies.json",
     "R_tcp_counterfactual", "collision_tcp_counterfactual"),
    ("AD-MLP", "dataset/results_v1/admlp_counterfactual_case_studies_aligned.json",
     "R_admlp_counterfactual", "collision_admlp_counterfactual"),
    ("UniAD", "dataset/results_v1/uniad_counterfactual_case_studies.json",
     "R_uniad_counterfactual", "collision_uniad_counterfactual"),
    ("VAD", "dataset/results_v1/vad_counterfactual_case_studies.json",
     "R_vad_counterfactual", "collision_vad_counterfactual"),
]


def load_means():
    rows = []
    for label, path, r_key, c_key in SOURCES:
        data = json.load(open(path))
        r_vals = [row[r_key] for row in data]
        c_vals = [row[c_key] for row in data]
        mean_r = sum(r_vals) / len(r_vals)
        coll_rate = sum(c_vals) / len(c_vals)
        rows.append((label, mean_r, coll_rate))
    return rows


def render(rows, out_path: Path):
    colors = {"TCP": "#4C4C4C", "AD-MLP": "#7F7F7F",
              "UniAD": "#C44E52", "VAD": "#4C72B0"}

    r_vals = [r for _, r, _ in rows]
    c_vals = [c for _, _, c in rows]
    r_lo, r_hi = min(r_vals), max(r_vals)
    c_lo, c_hi = min(c_vals), max(c_vals)
    pad = 0.12  # fractional padding so labels near the extremes don't clip

    def norm(v, lo, hi):
        # low value (safer) -> near y=1 (top); high value -> near y=0
        span = (hi - lo) or 1.0
        frac = (v - lo) / span
        return (1 - frac) * (1 - 2 * pad) + pad

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for label, r, c in rows:
        y_left = norm(r, r_lo, r_hi)
        y_right = norm(c, c_lo, c_hi)
        color = colors[label]
        lw = 2.2 if label in ("UniAD", "VAD") else 1.4
        ax.plot([0, 1], [y_left, y_right], color=color, linewidth=lw,
                marker="o", markersize=4, zorder=3 if label in ("UniAD", "VAD") else 2)
        ax.annotate(f"{label}  {r:.3f}", xy=(0, y_left), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=color)
        ax.annotate(f"{label}  {c*100:.0f}%", xy=(1, y_right), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=color)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(1, color="black", linewidth=0.8)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mean $R$\n(safer $\\to$ top)", "Coll. rate\n(safer $\\to$ top)"])
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    rows = load_means()
    for label, r, c in rows:
        print(f"{label:8s} mean_R={r:.3f}  coll_rate={c*100:.1f}%")
    render(rows, FIG_DIR / "planner_slope_chart.png")


if __name__ == "__main__":
    main()
