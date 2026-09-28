#!/usr/bin/env python3
"""Render root.tex fig:qualitative_per_timestep_comparison: a 2-row
(one real DeepAccident scenario per row) x 4-column (one method per column --
FutureRisk/CARM, RSS, raw 3D-TTC, GSSM) bird's-eye-view grid, each panel a
top-down ego path colored along its length by that method's own per-frame
danger score (green=low, red=high), with the interacting agent's path drawn
for context and the real collision contact (missed-collision row) or
closest-approach moment (near-miss row) marked explicitly.

Landscape orientation (rows=scenarios, columns=methods) replaces an earlier
portrait layout (rows=methods, columns=scenarios): the underlying real
trajectories are individually narrow/tall (`aspect="equal"` on real
bird's-eye geometry), so tiling 4 of them side-by-side per row uses the
page's horizontal room far better than stacking 4 of them in one column did
-- this is the whole reason for the switch, not merely a cosmetic reflow.

Consumes ONLY `dataset/results_v1/qualitative_per_timestep_comparison.json`
(written by `qualitative_per_timestep_comparison.py`, which does all the
real per-instance search + 4-method per-frame scoring; this script does no
scoring of its own, only geometry transforms + drawing).

Color-scale choice (documented here since it is a deliberate, non-obvious
call -- see the analysis script's own module docstring for why the four
methods' raw scores are NOT naturally on one shared scale):

  - FutureRisk (CARM), RSS, raw 3D-TTC are all, by construction, bounded to
    [0,1] (sigmoid / clip-and-normalize outputs) and genuinely comparable
    on that shared numeric scale -- these three columns use one FIXED
    Normalize(0, 1) with the same colormap in both rows, so a given color
    means the same raw score everywhere across those 6 panels.
  - GSSM's "intensity" (log10 of an equivalent close-encounter draw count,
    see `futurerisk/baselines_gssm.py`) is NOT bounded to [0,1] -- it is a
    different unit entirely, and its typical range differs a lot between
    a quiet scenario and a violent one. Forcing it onto the same [0,1]
    scale as the other three would either saturate it solid-red everywhere
    (if its raw values commonly exceed 1) or wash it out to solid green
    (if they sit near 0) -- neither conveys anything about ITS OWN
    within-trajectory variation. So the GSSM column is normalized per row
    (per scenario), to that scenario's own [min, max] observed over the 41
    frames shown -- i.e. GSSM's color in this figure means "how extreme,
    relative to this method's own range on this specific trajectory", not
    an absolute score comparable to the other three columns' colors. This
    is called out again directly on the figure via each GSSM panel's own
    colorbar (labeled with its actual value range) instead of a shared 0-1
    colorbar.

Every other visual choice (local ego-centered/heading-aligned frame per
panel, box outlines at t=0 (light) drawn for context, thin gray agent
path) follows `render_uniad_vs_vad_case_study.py`'s established bird's-eye
convention in this project.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.geometry import rect_corners

RESULTS_PATH = Path("dataset/results_v1/qualitative_per_timestep_comparison.json")
FIG_DIR = Path("icra/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "qualitative_per_timestep_comparison.png"

plt.rcParams.update({"font.size": 8})

METHOD_COLS = [
    ("futurerisk_carm", "FutureRisk (CARM)"),
    ("raw_ttc", "Raw 3D-TTC"),
    ("rss", "RSS"),
    ("gssm", "GSSM"),
]
SCENARIO_ROWS = [
    ("near_miss", "Near-miss"),
    ("missed_collision", "Collision"),
]
CMAP = plt.get_cmap("RdYlGn_r")  # green = low risk, red = high risk


def _to_local_frame(ego, agent):
    """Same local-frame convention as
    `render_uniad_vs_vad_case_study.py::_to_local_frame`: origin at ego's
    own t=0 position, ego's own t=0 heading rotated to point along +y."""
    x0, y0 = ego["x"][0], ego["y"][0]
    theta0 = ego["heading"][0] - np.pi / 2

    def rot(px, py):
        dx, dy = np.asarray(px) - x0, np.asarray(py) - y0
        c, s = np.cos(-theta0), np.sin(-theta0)
        return c * dx - s * dy, s * dx + c * dy

    ego_lx, ego_ly = rot(ego["x"], ego["y"])
    agent_lx, agent_ly = rot(agent["x"], agent["y"])
    ego_lh = np.asarray(ego["heading"]) - theta0
    agent_lh = np.asarray(agent["heading"]) - theta0
    return ego_lx, ego_ly, ego_lh, agent_lx, agent_ly, agent_lh


def _draw_box(ax, cx, cy, heading, length, width, color, alpha=1.0, zorder=3, edgecolor="black", lw=0.5):
    corners = rect_corners(np.array([cx]), np.array([cy]), np.array([heading]),
                            np.array([length]), np.array([width]))[0]
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor=edgecolor,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(poly)


def _colored_path(ax, lx, ly, scores, norm, cmap, linewidth=2.2, zorder=3):
    """Draw the ego path as a LineCollection colored per-SEGMENT by the
    score at that segment's start frame -- the standard matplotlib
    "color a line by a scalar along its length" recipe."""
    points = np.stack([lx, ly], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=zorder)
    lc.set_array(np.asarray(scores[:-1]))
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)
    return lc


def render_panel(ax, entry: dict, method_key: str, norm, show_xlabel: bool, show_ylabel: bool,
                  label_fontsize_scale: float = 1.0):
    ego, agent = entry["ego_box"], entry["agent_box"]
    ego_lx, ego_ly, ego_lh, agent_lx, agent_ly, agent_lh = _to_local_frame(ego, agent)
    scores = np.asarray(entry["scores"][method_key])

    # Interacting agent's path: thin, faded gray, for context only (not
    # itself risk-colored -- only ego's own path carries each method's
    # score, per the task's "color ego's path" spec).
    ax.plot(agent_lx, agent_ly, color="#888888", linewidth=1.0, alpha=0.55, zorder=1, linestyle="--")

    lc = _colored_path(ax, ego_lx, ego_ly, scores, norm, CMAP)

    # Light context boxes at t=0 for both vehicles.
    _draw_box(ax, ego_lx[0], ego_ly[0], ego_lh[0], ego["length"][0], ego["width"][0],
              "#4C72B0", alpha=0.20, zorder=2)
    _draw_box(ax, agent_lx[0], agent_ly[0], agent_lh[0], agent["length"][0], agent["width"][0],
              "#888888", alpha=0.20, zorder=2)

    # Mark the actual NATIVE-CONFIRMED collision contact (missed-collision
    # row only) or the closest-approach moment (near-miss row) explicitly,
    # on every panel (this is ground truth / the closed-form margin, not a
    # per-method quantity, so it is drawn identically regardless of which
    # method's score colors the path). Gated on `native_collision_label`,
    # NOT merely `collision_frame_index is not None`: the near-miss
    # instance's own highlighted agent DOES have a `collision_frame_index`
    # too (a real geometric SAT box-touch fires at its worst-margin frame,
    # see `qualitative_per_timestep_comparison.py`'s `_instance_stat_row`
    # docstring) even though native ground truth confirms no actual
    # collision -- exactly the point of that row, so it must get the
    # "closest approach" marker, not an "actual collision" marker.
    coll_idx = entry.get("collision_frame_index") if entry.get("native_collision_label") == 1 else None
    if coll_idx is not None:
        ax.plot(ego_lx[coll_idx], ego_ly[coll_idx], marker="X", markersize=9, color="black",
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        _draw_box(ax, ego_lx[coll_idx], ego_ly[coll_idx], ego_lh[coll_idx], ego["length"][coll_idx], ego["width"][coll_idx],
                  "none", alpha=1.0, zorder=5, edgecolor="black", lw=1.1)
        _draw_box(ax, agent_lx[coll_idx], agent_ly[coll_idx], agent_lh[coll_idx],
                  agent["length"][coll_idx], agent["width"][coll_idx], "none", alpha=1.0, zorder=5,
                  edgecolor="black", lw=1.1)
    else:
        idx = entry["closest_approach_frame_index"]
        ax.plot(ego_lx[idx], ego_ly[idx], marker="o", markersize=6, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=1.1, zorder=6)

    # adjustable="box" (not "datalim"): these real trajectories are much
    # longer along y (heading direction) than x, so "datalim" -- which
    # pads the DATA LIMITS to force equal aspect within whatever box shape
    # the layout engine happens to give this subplot -- inflates the
    # x-range to dozens of extra empty meters on each side. "box" instead
    # shrinks the AXES BOX itself to the data's true aspect ratio, giving a
    # tight, correctly-scaled bird's-eye view letterboxed within the grid
    # cell -- and with 4 methods now tiled per row (landscape layout) that
    # letterboxing costs far less wasted horizontal room than the old
    # 2-wide portrait grid did.
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))  # these narrow, tall
    # boxes have too little horizontal room for a default (5-6 tick) axis --
    # ticks collide into unreadable text otherwise. At the 1.5x enlarged
    # label size used for the standalone per-method crops, even 3 tick
    # labels collide ("-3"/"0" overlap into "-30"); dropping the middle tick
    # (keeping just the two extremes, which is all that's needed to read the
    # panel's x-scale) removes the collision without touching the composite
    # grid figure, which keeps all 3.
    if label_fontsize_scale > 1.0:
        xlim = ax.get_xlim()
        xticks = [t for t in ax.xaxis.get_majorticklocs() if xlim[0] <= t <= xlim[1]]
        if len(xticks) > 2:
            ax.set_xticks([xticks[0], xticks[-1]])
    if show_xlabel:
        ax.set_xlabel("local x (m)", fontsize=6.5 * label_fontsize_scale)
    if show_ylabel:
        ax.set_ylabel("local y (m)", fontsize=6.5 * label_fontsize_scale)
    ax.tick_params(labelsize=5.5 * label_fontsize_scale)
    return lc


def _gssm_norm(entry):
    """Per-scenario GSSM normalization -- shared by the composite figure and
    the standalone per-method crops so both use exactly the same color
    mapping (see module docstring: GSSM's unbounded units are normalized to
    each scenario's own observed [min, max])."""
    gssm_scores = np.asarray(entry["scores"]["gssm"])
    return matplotlib.colors.Normalize(vmin=float(gssm_scores.min()), vmax=float(gssm_scores.max()))


def _short_id(entry):
    town = entry["instance_id"].split("_", 1)[0]
    route = entry["instance_id"].rsplit("__", 1)[-1]
    return f"{town} · {route}"


def render_single_method(method_key: str, method_label: str, instances: dict, fixed_norm) -> Path:
    """Save one method's own standalone 2-row (near-miss / collision) x
    1-column figure -- the same panels as that method's column in the
    composite grid, cropped out into an independent file per the user's
    request ("把每一个metrics的小图截取出来单独储存一下"), rather than a
    pixel-crop of the composite PNG, so headers/row-labels/colorbars are
    freshly laid out and self-contained in each file."""
    # Text scale for these standalone per-method crops only (composite grid
    # in `main()` is unaffected -- `render_panel`'s default
    # `label_fontsize_scale=1.0` there): all non-title text (axis
    # labels/ticks, colorbar ticks/labels, row labels, instance ids) is
    # enlarged 1.5x for readability when these files are used standalone;
    # the method-name title (`fig.text(... method_label ...)` below) is
    # deliberately excluded per request.
    TEXT_SCALE = 1.5

    n_rows = len(SCENARIO_ROWS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(2.5, 5.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.03)

    row_headers = []
    for row_idx, (row_key, row_label) in enumerate(SCENARIO_ROWS):
        entry = instances[row_key]
        norm = _gssm_norm(entry) if method_key == "gssm" else fixed_norm
        ax = axes[row_idx]
        lc = render_panel(ax, entry, method_key, norm,
                           show_xlabel=(row_idx == n_rows - 1), show_ylabel=True,
                           label_fontsize_scale=TEXT_SCALE)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=5 * TEXT_SCALE)
        if method_key == "gssm":
            cbar.set_label("GSSM intensity\n(this trajectory's own range)", fontsize=5 * TEXT_SCALE)
        else:
            cbar.set_label("score (0-1)", fontsize=5 * TEXT_SCALE)
        row_headers.append((row_label, _short_id(entry)))

    top_margin_in = 0.34
    left_margin_in = 0.80
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig.get_layout_engine().set(rect=(
        left_margin_in / fig_w_in, 0.0,
        1 - left_margin_in / fig_w_in, 1 - top_margin_in / fig_h_in,
    ))
    fig.canvas.draw()
    TOP = 1 - top_margin_in / fig_h_in

    # Centered on the FULL figure width (0.5), not the narrow panel's own
    # center: with only one column per standalone file (unlike the
    # composite grid, which needs per-column centering to match each
    # column's position), centering the title on the panel instead left it
    # clipping past the right edge of the canvas for the longer method
    # names (e.g. "FutureRisk (CARM)").
    fig.text(0.5, TOP + 0.010, method_label,
              ha="center", va="bottom", fontsize=9, fontweight="bold")

    for row_idx, (row_label, inst_id) in enumerate(row_headers):
        pos = axes[row_idx].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        x_label = left_margin_in / fig_w_in * 0.30
        fig.text(x_label, y_center, row_label, ha="center", va="center",
                  fontsize=8 * TEXT_SCALE, fontweight="bold", rotation=90)
        fig.text(x_label + left_margin_in / fig_w_in * 0.34, y_center, inst_id,
                  ha="center", va="center", fontsize=5.8 * TEXT_SCALE, rotation=90)

    slug = method_key
    out_path = FIG_DIR / f"qualitative_per_timestep_comparison_{slug}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    instances = results["instances"]
    n_rows, n_cols = len(SCENARIO_ROWS), len(METHOD_COLS)
    # constrained_layout (not tight_layout): tight_layout only trims the
    # OUTER margin around the union of axes bounding boxes, so when
    # aspect="equal" shrinks each individual panel to a narrow tall box
    # (these are real, narrow/tall trajectories -- see the module
    # docstring), the now-oversized *nominal* grid cells tight_layout
    # leaves in place still reserve their original width, showing up as
    # large empty margins. constrained_layout's solver instead re-packs
    # each axes' actual (post-aspect-lock) box tightly against its
    # neighbors, which is exactly the "many aspect-locked narrow panels"
    # case it is designed for.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.2, 5.0), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03)

    # CARM/RSS/raw-3D-TTC: fixed, shared [0,1] normalization (all three are
    # bounded-by-construction risk-like scores, see module docstring).
    # PowerNorm(gamma=0.45), not linear: this is a perceptual contrast
    # stretch only -- it does not change any underlying score, only how a
    # given score maps to a color. gamma<1 spreads out the LOW end of the
    # [0,1] range (e.g. 0.0 vs 0.2 vs 0.5 become much more visually
    # distinct) at the cost of compressing the high end (0.7 vs 1.0 look
    # more similar) -- appropriate here since the real, honestly-computed
    # scores in this figure mostly sit in the low-to-moderate range (see
    # qualitative_per_timestep_comparison.py's own selection stats), so a
    # linear map was washing out real differences that matter (e.g.
    # raw-3D-TTC's genuine flat 0.0 vs. CARM's genuine 0.85 peak looked
    # like a mild color difference under linear scaling; they are visually
    # obviously different under this stretch, with no number changed).
    fixed_norm = matplotlib.colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)

    row_headers = []  # (row_label, short_id) per row, for left-side row labels
    top_row_cbar_ax = [None] * n_cols  # row-0 colorbar Axes per column, for header centering below
    for row_idx, (row_key, row_label) in enumerate(SCENARIO_ROWS):
        entry = instances[row_key]
        # GSSM's own per-row (per-scenario) normalization (unbounded units,
        # see docstring).
        gssm_norm = _gssm_norm(entry)

        for col_idx, (method_key, method_label) in enumerate(METHOD_COLS):
            ax = axes[row_idx, col_idx]
            norm = gssm_norm if method_key == "gssm" else fixed_norm
            lc = render_panel(
                ax, entry, method_key, norm,
                show_xlabel=(row_idx == n_rows - 1), show_ylabel=(col_idx == 0),
            )
            cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=5)
            # For the 3 shared-scale columns the label is identical in
            # both rows, so it is only drawn once (top row) to save space;
            # GSSM's scale differs per row (per-scenario normalization),
            # so its label must appear on every row.
            if method_key == "gssm":
                cbar.set_label("GSSM intensity\n(this trajectory's own range)", fontsize=5)
            elif row_idx == 0:
                cbar.set_label("score (0-1)", fontsize=5)
            if row_idx == 0:
                top_row_cbar_ax[col_idx] = cbar.ax

        # Full instance_id strings (e.g.
        # "Town06_type002_subtype0001_000002__other_vehicle__r0060") are too
        # long to show in full without overlapping; show a short,
        # human-readable town+route identifier here instead (the full id,
        # and native_collision_label/n_agents, are still in the results
        # JSON for reproducibility -- dropped from the figure itself at the
        # user's request to keep headers minimal).
        row_headers.append((row_label, _short_id(entry)))

    # With constrained_layout, subplot positions are only finalized once
    # the solver actually runs (at a draw), so an axes' `.get_position()`
    # immediately after creating it is stale -- force one draw first to get
    # each panel's real, tightly-packed final box, then place header/row
    # label text just outside that (leaving small explicit margins for
    # them, added back on top of the constrained axes grid via `rect`
    # rather than reserved through wasted subplot space).
    top_margin_in = 0.42   # column (method) headers only -- one short line
    left_margin_in = 0.62  # row (scenario) labels, rotated, to the left
    bottom_margin_in = 0.30  # one-line caption
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig.get_layout_engine().set(rect=(
        left_margin_in / fig_w_in, bottom_margin_in / fig_h_in,
        1 - left_margin_in / fig_w_in, 1 - (top_margin_in + bottom_margin_in) / fig_h_in,
    ))
    fig.canvas.draw()
    TOP = 1 - top_margin_in / fig_h_in
    BOTTOM = bottom_margin_in / fig_h_in

    # Column (method) headers as figure-level text, placed AFTER the forced
    # draw above so each header uses its column's actual finalized
    # horizontal center (an axes-fraction annotation drifted out of
    # alignment here once the layout engine repositioned these unusually
    # narrow, tall trajectory subplots in earlier iterations of this
    # figure). Center each header over the FULL column -- the panel plus
    # its (top-row) colorbar -- not just the bare panel axes; the colorbar
    # sitting to the right of each panel shifts the column's true visual
    # center right of the panel's own center.
    for col_idx, (method_key, method_label) in enumerate(METHOD_COLS):
        pos_panel = axes[0, col_idx].get_position()
        pos_cbar = top_row_cbar_ax[col_idx].get_position()
        x_center = (pos_panel.x0 + pos_cbar.x1) / 2
        fig.text(x_center, TOP + 0.012, method_label, ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold")

    # Row (scenario) labels, rotated 90 deg, to the left of each row --
    # vertically centered on that row's actual finalized box (spans from
    # the bottom-most to top-most panel in the row, across all 4 columns,
    # since panel heights can differ slightly row to row).
    for row_idx, (row_label, inst_id) in enumerate(row_headers):
        y0s = [axes[row_idx, c].get_position().y0 for c in range(n_cols)]
        y1s = [axes[row_idx, c].get_position().y1 for c in range(n_cols)]
        y_center = (min(y0s) + max(y1s)) / 2
        x_label = left_margin_in / fig_w_in * 0.30
        fig.text(x_label, y_center, row_label, ha="center", va="center",
                  fontsize=8, fontweight="bold", rotation=90)
        fig.text(x_label + left_margin_in / fig_w_in * 0.34, y_center, inst_id,
                  ha="center", va="center", fontsize=5.8, rotation=90)

    fig.text(
        0.5, BOTTOM - 0.01,
        "4 methods (columns) x 2 real scenarios (rows); path color = that method's score "
        "(green=low, red=high). Gray dashed = other agent; X = actual collision; o = closest approach.",
        ha="center", va="top", fontsize=6.3,
    )

    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")

    # Also save each method's column as its own standalone figure (same
    # panels, freshly laid out with their own header/row-labels/colorbars),
    # per the user's request to have each method's small figure available
    # individually rather than only embedded in the composite grid.
    for method_key, method_label in METHOD_COLS:
        out = render_single_method(method_key, method_label, instances, fixed_norm)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
