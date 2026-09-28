"""Case-study visualization for qualitative validation (S3.11).

Plots R(t) (or m(t) and per-bin hazard q_k) over time for a handful of
collision-free trajectory instances, to visually demonstrate margin
collapse and recovery that a binary collision label would not capture.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe default; caller can switch backends before import if needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_margin_and_risk_over_time(
    times: np.ndarray,
    margin_t: np.ndarray,
    risk_t: np.ndarray,
    title: str = "",
    collision_free: bool = True,
    ax=None,
):
    """Single-instance plot: m(t) on one axis, r(t) on a twin axis, so a
    viewer can see a margin collapse-and-recover event that never crosses
    into collision (S3.11's core motivating visualization).

    m(t) is +inf wherever tau_3D never triggers within the TTC cap (S3.2) --
    a genuinely, maximally safe instant, not missing data. Plotting +inf
    directly breaks matplotlib's axis autoscaling (an all-infinite array
    silently collapses to a near-empty default-range plot), so infinities
    are clipped to a finite display cap purely for rendering; the
    underlying array/analysis is never modified.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    margin_t = np.asarray(margin_t, dtype=float)
    finite = margin_t[np.isfinite(margin_t)]
    any_inf = not np.all(np.isfinite(margin_t))
    display_cap = float(np.max(np.abs(finite))) * 1.5 if finite.size else 5.0
    display_cap = max(display_cap, 1.0)
    margin_display = np.clip(margin_t, -display_cap, display_cap)

    ax.plot(times, margin_display, color="tab:blue", label="margin m(t) [s]")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("margin m(t) [s]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    if any_inf:
        ax.text(
            0.01, 0.02, "margin capped for display; true value may be +/-inf",
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom", ha="left",
        )

    ax2 = ax.twinx()
    ax2.plot(times, risk_t, color="tab:red", label="risk r(t)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("risk r(t)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    status = "collision-free" if collision_free else "collision"
    ax.set_title(f"{title} ({status})" if title else status)
    fig.tight_layout()
    return fig, (ax, ax2)


def plot_hazard_bins(
    bin_centers: np.ndarray, q_k: np.ndarray, title: str = "", ax=None,
):
    """Bar plot of the per-bin predicted hazard q_k^j (S3.11 alternative:
    "visualize m(t) and per-bin hazard q_k")."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2))
    else:
        fig = ax.figure
    ax.bar(bin_centers, q_k, width=(bin_centers[1] - bin_centers[0]) * 0.9 if len(bin_centers) > 1 else 0.2,
           color="tab:orange")
    ax.set_ylim(0, max(1e-3, q_k.max() * 1.2))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("hazard q_k")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def case_study_grid(
    instances: Sequence[dict], save_path: Optional[str] = None,
):
    """Grid of case-study plots for several instances, ideally chosen with
    visibly different outcomes despite both being collision-free (S3.11):
    e.g. one with a large sustained margin, one with a brief close call.

    Each entry in `instances` is a dict with keys: 'times', 'margin_t',
    'risk_t', 'bin_centers' (optional), 'q_k' (optional), 'title'
    (optional), 'collision_free' (optional, default True).
    """
    n = len(instances)
    has_hazard = any("q_k" in inst for inst in instances)
    rows_per_instance = 2 if has_hazard else 1
    fig, axes = plt.subplots(rows_per_instance * n, 1, figsize=(7, 3 * rows_per_instance * n), squeeze=False)

    for i, inst in enumerate(instances):
        row = i * rows_per_instance
        plot_margin_and_risk_over_time(
            inst["times"], inst["margin_t"], inst["risk_t"],
            title=inst.get("title", f"instance {i}"),
            collision_free=inst.get("collision_free", True),
            ax=axes[row][0],
        )
        if has_hazard and "q_k" in inst:
            plot_hazard_bins(inst["bin_centers"], inst["q_k"], ax=axes[row + 1][0])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def _colored_line(ax, x: np.ndarray, y: np.ndarray, c: np.ndarray, cmap: str, vmin: float, vmax: float,
                   linewidth: float = 3.0, zorder: int = 3):
    """Draw (x,y) as a sequence of segments each colored by the matching
    value in `c` (e.g. instantaneous risk) -- a single matplotlib Line2D
    can't vary color along its length, so this builds a LineCollection of
    per-step segments instead. Returns the LineCollection (for a colorbar)."""
    points = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax), zorder=zorder)
    lc.set_array(0.5 * (c[:-1] + c[1:]))  # per-segment color = avg of its two endpoints
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)
    return lc


def plot_bev_trajectory(
    ego_xy: np.ndarray,
    agent_xys: dict,
    risk_t: np.ndarray,
    R_predicted: Optional[float] = None,
    y_analytic: Optional[float] = None,
    collision_label: Optional[int] = None,
    title: str = "",
    max_agents: int = 4,
    ax=None,
):
    """Bird's-eye-view (top-down x/y) plot of one trajectory instance: ego's
    path colored along its length by the analytic instantaneous risk r(t)
    (S3.4's phi(m(t)), independent of the trained model -- a fixed,
    inspectable reference), with the closest few agents' paths overlaid for
    context, and the model's own predicted score (plus the analytic target
    and true collision label, if given) annotated in the title.

    This is the "does the score make geometric sense" check: a viewer can
    see the actual paths and where along them the color goes red, next to
    the number the model assigned to the whole 5s window.

    ego_xy: (T, 2). agent_xys: {agent_id: (T, 2)} -- pass only the agents
    worth showing (e.g. already sorted by closeness); at most `max_agents`
    are drawn, the rest are silently dropped to keep the panel readable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ego_xy = np.asarray(ego_xy, dtype=float)
    risk_t = np.asarray(risk_t, dtype=float)

    # Agent paths are deliberately muted/neutral (thin dashed gray-blue,
    # distinguished from each other only by marker shape) -- they used to
    # use bright tab10 colors, which at a glance could be mistaken for a
    # warm (high-risk) segment of the ego line itself, since both were
    # rendered as colored lines competing for the same visual channel. The
    # ego line's color is the only thing that encodes risk in this plot.
    agent_markers = ["o", "^", "D", "P", "X", "v"]
    agent_items = list(agent_xys.items())[:max_agents]
    for i, (aid, xy) in enumerate(agent_items):
        xy = np.asarray(xy, dtype=float)
        marker = agent_markers[i % len(agent_markers)]
        ax.plot(xy[:, 0], xy[:, 1], color="0.55", linewidth=1.0, alpha=0.7,
                linestyle="--", zorder=2)
        ax.scatter(xy[0, 0], xy[0, 1], facecolor="none", edgecolor="0.4", marker=marker, s=30, zorder=4,
                   linewidth=1.2, label=f"agent {aid} start/end")
        ax.scatter(xy[-1, 0], xy[-1, 1], color="0.4", marker=marker, s=30, zorder=4)

    lc = _colored_line(ax, ego_xy[:, 0], ego_xy[:, 1], risk_t, cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
                        linewidth=5.0, zorder=3)
    ax.scatter(ego_xy[0, 0], ego_xy[0, 1], color="black", marker="o", s=70, zorder=5, label="ego start",
               edgecolor="white", linewidth=0.8)
    ax.scatter(ego_xy[-1, 0], ego_xy[-1, 1], color="black", marker="^", s=80, zorder=5, label="ego end",
               edgecolor="white", linewidth=0.8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=6, loc="best", framealpha=0.7)
    fig.colorbar(lc, ax=ax, label="analytic risk r(t)", shrink=0.7)

    label_str = "" if collision_label is None else f"  collision={collision_label}"
    score_str = ""
    if R_predicted is not None:
        score_str += f"R_pred={R_predicted:.3f}"
    if y_analytic is not None:
        score_str += f"  y_analytic={y_analytic:.3f}"
    ax.set_title(f"{title}\n{score_str}{label_str}", fontsize=8)
    return fig, ax


def bev_grid(entries: Sequence[dict], n_cols: int = 3, save_path: Optional[str] = None):
    """Grid of `plot_bev_trajectory` panels, meant to span a range of
    predicted scores (e.g. sorted low -> high) so a viewer can compare
    trajectory shapes side by side against the numbers assigned to them.

    Each entry: dict with keys 'ego_xy', 'agent_xys', 'risk_t', and
    optionally 'R_predicted', 'y_analytic', 'collision_label', 'title'.
    """
    n = len(entries)
    n_cols = max(1, min(n_cols, n))
    n_rows = -(-n // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for i, entry in enumerate(entries):
        r, c = divmod(i, n_cols)
        plot_bev_trajectory(
            entry["ego_xy"], entry["agent_xys"], entry["risk_t"],
            R_predicted=entry.get("R_predicted"), y_analytic=entry.get("y_analytic"),
            collision_label=entry.get("collision_label"), title=entry.get("title", f"instance {i}"),
            max_agents=entry.get("max_agents", 4), ax=axes[r][c],
        )
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
