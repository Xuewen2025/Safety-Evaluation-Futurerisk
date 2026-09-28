#!/usr/bin/env python3
"""Analysis + figure for root.tex Sec. exp_tradeoff, review item P0-1: turn
the single scalar "Anchored 70.1%" (Table interpretability_tradeoff) into a
distribution, read from the per-agent CSV written by
`v1_interpretability.py` (dataset/results_v1/v1_interpretability_per_agent.csv).

Adds two things beyond what v1_interpretability.py already reports:
  1. A rank-biserial effect size alongside each Mann-Whitney p-value (with
     n in the tens of thousands, p-values alone are close to meaningless --
     see e.g. collision vs. no-collision below, p=6.5e-36 but a median gap
     of only 0.009).
  2. An explicit accounting of the extreme-outlier tail: `anchored_frac =
     1 - |beta*tanh(g_r)| / |logit(R_tilde)|` is unbounded below whenever
     R_tilde sits close to the structured branch's own decision boundary
     (R_tilde ~ 0.5, logit ~ 0) -- a small population of agents (<1%) where
     this ratio blows up, dragging the MEAN (70.1%) below the MEDIAN. This
     is reported honestly as a known artifact of this specific ratio-based
     operationalization, not a second interpretability regime, and a
     winsorized median/IQR (excluding that tail) is reported alongside the
     raw one so a reader can see the headline number is not sensitive to it.

No new model inference: reads only the already-written per-agent CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

RESULTS_DIR = Path("dataset/results_v1")
FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 13.5})

# Below this, anchored_frac reflects the ratio-blowup artifact (R_tilde's own
# logit near 0), not a meaningfully "less anchored" agent -- see module
# docstring. Used only to report an outlier-excluded robustness check; the
# raw distribution (all agents, no filtering) is still the primary evidence.
OUTLIER_FLOOR = -1.0


def _load(dataset: str):
    rows = []
    with open(RESULTS_DIR / "v1_interpretability_per_agent.csv") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset:
                rows.append({
                    "anchored_frac": float(row["anchored_frac"]),
                    "native_collision_label": row["native_collision_label"] == "1",
                    "critical_event_triggered": row["critical_event_triggered"] == "True",
                })
    return rows


def _rank_biserial(a: np.ndarray, b: np.ndarray, u_stat: float) -> float:
    """Rank-biserial correlation for Mann-Whitney U, r = 1 - 2U/(n1*n2),
    in [-1, 1]; |r| small means the two groups' distributions overlap
    almost completely regardless of how small the p-value is."""
    return float(1.0 - (2.0 * u_stat) / (len(a) * len(b)))


def _describe(vals: np.ndarray) -> dict:
    return {
        "n": int(len(vals)), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)), "q75": float(np.percentile(vals, 75)),
    }


def analyze(dataset: str) -> dict:
    rows = _load(dataset)
    all_vals = np.array([r["anchored_frac"] for r in rows])
    coll = np.array([r["anchored_frac"] for r in rows if r["native_collision_label"]])
    noncoll = np.array([r["anchored_frac"] for r in rows if not r["native_collision_label"]])
    evt = np.array([r["anchored_frac"] for r in rows if r["critical_event_triggered"]])
    noevt = np.array([r["anchored_frac"] for r in rows if not r["critical_event_triggered"]])

    u_coll, p_coll = sstats.mannwhitneyu(coll, noncoll, alternative="two-sided")
    u_evt, p_evt = sstats.mannwhitneyu(evt, noevt, alternative="two-sided")

    outlier_mask = all_vals < OUTLIER_FLOOR
    winsorized = all_vals[~outlier_mask]

    return {
        "dataset": dataset,
        "overall_raw": _describe(all_vals),
        "overall_winsorized_excl_outlier_tail": _describe(winsorized),
        "outlier_tail": {
            "floor": OUTLIER_FLOOR, "n_below_floor": int(outlier_mask.sum()),
            "pct_below_floor": float(100.0 * outlier_mask.mean()),
            "min_value": float(all_vals.min()),
        },
        "native_collision": _describe(coll), "no_collision": _describe(noncoll),
        "collision_vs_no_collision": {
            "mannwhitney_p": float(p_coll),
            "rank_biserial_r": _rank_biserial(coll, noncoll, u_coll),
        },
        "critical_event_triggered": _describe(evt), "no_critical_event": _describe(noevt),
        "critical_event_vs_none": {
            "mannwhitney_p": float(p_evt),
            "rank_biserial_r": _rank_biserial(evt, noevt, u_evt),
        },
    }


def render_figure(da_rows, bd_rows):
    """3-panel boxplot (overall / by collision label / by critical-event
    trigger), DeepAccident in-domain (primary) and Bench2Drive cross-domain
    (secondary) side by side. Clipped display range [-1, 1] -- the rare
    (<1%) sub-floor outlier tail is annotated in the caption, not shown, so
    the box/whisker geometry stays legible for the >99% of agents inside it."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    panels = [
        ("Overall", lambda rows: [[r["anchored_frac"] for r in rows]], ["all agents"]),
        ("By collision label", lambda rows: [
            [r["anchored_frac"] for r in rows if not r["native_collision_label"]],
            [r["anchored_frac"] for r in rows if r["native_collision_label"]],
        ], ["no collision", "collision"]),
        ("By critical event", lambda rows: [
            [r["anchored_frac"] for r in rows if not r["critical_event_triggered"]],
            [r["anchored_frac"] for r in rows if r["critical_event_triggered"]],
        ], ["no event", "triggered"]),
    ]
    colors = {"DeepAccident (in)": "#4C72B0", "Bench2Drive (cross)": "#DD8452"}
    for ax, (title, split_fn, sublabels) in zip(axes, panels):
        da_groups = split_fn(da_rows)
        bd_groups = split_fn(bd_rows)
        data, positions, box_colors, xticklabels = [], [], [], []
        pos = 0
        for lbl, da_g, bd_g in zip(sublabels, da_groups, bd_groups):
            for tag, g in [("DeepAccident (in)", da_g), ("Bench2Drive (cross)", bd_g)]:
                data.append(np.clip(g, -1.0, 1.0))
                positions.append(pos)
                box_colors.append(colors[tag])
                pos += 1
            xticklabels.append(lbl)
            pos += 0.6
        bp = ax.boxplot(data, positions=positions, widths=0.75, patch_artist=True,
                         showfliers=False, medianprops={"color": "black", "linewidth": 1.2})
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        tick_pos = [np.mean(positions[i * 2:i * 2 + 2]) for i in range(len(xticklabels))]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(xticklabels, fontsize=10.5, rotation=15, ha="right")
        ax.set_title(title, fontsize=12)
        ax.axhline(0.701, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("anchored fraction (per agent)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.8) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="lower center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    out = FIG_DIR / "anchored_fraction_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15,
                bbox_extra_artists=[axes[0].yaxis.label])
    plt.close(fig)
    print(f"wrote {out}")


def main():
    da_rows = _load("deepaccident")
    bd_rows = _load("bench2drive")
    results = {"deepaccident_indomain": analyze("deepaccident"), "bench2drive_crossdomain": analyze("bench2drive")}
    out_path = RESULTS_DIR / "anchored_fraction_distribution_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    for k, v in results.items():
        print(f"\n=== {k} ===")
        print(f"  overall: mean={v['overall_raw']['mean']:.3f} median={v['overall_raw']['median']:.3f} "
              f"IQR=[{v['overall_raw']['q25']:.3f},{v['overall_raw']['q75']:.3f}] n={v['overall_raw']['n']}")
        print(f"  outlier tail (< {OUTLIER_FLOOR}): {v['outlier_tail']['n_below_floor']} agents "
              f"({v['outlier_tail']['pct_below_floor']:.3f}%), min={v['outlier_tail']['min_value']:.1f}")
        print(f"  winsorized (excl. tail): mean={v['overall_winsorized_excl_outlier_tail']['mean']:.3f} "
              f"median={v['overall_winsorized_excl_outlier_tail']['median']:.3f}")
        print(f"  collision vs no-collision: median {v['native_collision']['median']:.3f} vs "
              f"{v['no_collision']['median']:.3f}, p={v['collision_vs_no_collision']['mannwhitney_p']:.2e}, "
              f"rank-biserial r={v['collision_vs_no_collision']['rank_biserial_r']:.3f}")
        print(f"  critical-event vs none: median {v['critical_event_triggered']['median']:.3f} vs "
              f"{v['no_critical_event']['median']:.3f}, p={v['critical_event_vs_none']['mannwhitney_p']:.2e}, "
              f"rank-biserial r={v['critical_event_vs_none']['rank_biserial_r']:.3f}")
    render_figure(da_rows, bd_rows)


if __name__ == "__main__":
    main()
