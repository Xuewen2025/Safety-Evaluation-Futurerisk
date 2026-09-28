#!/usr/bin/env python3
"""Render the root.tex figures for which we have real (not placeholder) data:
  - fig:residual_distribution (g_r by interaction type)
  - fig:severity_bins (R across a(t) bins, collision-free instances)
using matplotlib, saved under icra/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = Path("dataset/results_v1")
FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9})


def render_residual_distribution():
    with open(RESULTS_DIR / "v1_interpretability_summary.json") as f:
        d = json.load(f)
    by_type = d["g_r_by_type_deepaccident"]
    order = ["car-following", "crossing", "merging/lane-change", "oncoming", "other"]
    order = [t for t in order if t in by_type]
    means = [by_type[t]["mean"] for t in order]
    stds = [by_type[t]["std"] for t in order]
    ns = [by_type[t]["n"] for t in order]

    fig, ax = plt.subplots(figsize=(4.3, 2.6))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(order, ns)], fontsize=7)
    ax.set_ylabel(r"$g_r(\mathbf{h}^j)$ (raw, pre-tanh)")
    ax.set_title("Residual by interaction type (DeepAccident in-domain)")
    fig.tight_layout()
    out = FIG_DIR / "residual_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def render_severity_bins():
    with open(RESULTS_DIR / "v1_full_eval_summary.json") as f:
        d = json.load(f)
    bins = d["severity_bins"]["max_evasive_severity"]
    if not bins:
        print("no severity_bins data for max_evasive_severity")
        return
    x = [b["bin"] for b in bins]
    r_mean = [b["R_predicted_mean"] for b in bins]
    r_median = [b["R_predicted_median"] for b in bins]
    labels = [f"Q{b['bin'] + 1}" for b in bins]

    fig, ax = plt.subplots(figsize=(4.3, 2.6))
    ax.plot(x, r_mean, "o-", label="mean $R$", color="#C44E52")
    ax.plot(x, r_median, "s--", label="median $R$", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$a(t)$ quintile")
    ax.set_ylabel("$R$")
    ax.legend(fontsize=7)
    ax.set_title(r"$R$ vs. evasive-severity quintile ($\rho=0.276$)")
    fig.tight_layout()
    out = FIG_DIR / "severity_bins.png"
    fig.savefig(out, bbox_inches="tight")
    out_hires = FIG_DIR / "severity_bins_hires.png"
    fig.savefig(out_hires, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {out_hires}")


if __name__ == "__main__":
    render_residual_distribution()
    render_severity_bins()
