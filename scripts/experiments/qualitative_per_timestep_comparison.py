#!/usr/bin/env python3
"""Per-timestep (frame-by-frame) 4-method danger-score comparison on two
real DeepAccident instances, for root.tex Sec.~IV-C's qualitative figure
(fig:qualitative_per_timestep_comparison). Produces the results JSON that
`render_qualitative_per_timestep_comparison.py` renders into the 2x4
bird's-eye grid; this script does NOT render anything.

Two real, searched (not fabricated) instances
----------------------------------------------
1. **Successfully-avoided near-miss**: native ground truth says NO
   collision (`native_collision_label == 0`), but the closed-form
   geometric margin `m(t) = tau_3D(t) - tau_phy(t)` (eq:margin) goes
   genuinely negative at some frame (a real projected-overlap /
   safe-TTC-violation event that never actually results in a logged
   collision -- i.e. an avoided close call, not a quiet drive).
2. **Collision missed by the geometric criterion**: native ground truth
   CONFIRMS a real collision (`native_collision_label == 1`, i.e. `y` was
   native-anchored per Sec.~III-C.5 /
   `futurerisk/adapters/deepaccident.py`'s `native_is_collision` +
   plausibility-guarded flag patch), but the worst-case (min-over-agents)
   3D-TTC `tau_3D(t)` never drops below the forward-simulation cap at ANY
   frame of the instance -- i.e. `raw_ttc_baseline_score` stays at (or
   extremely close to) 0 throughout, the concrete "geometric criterion
   misses the large majority of true collisions outright" failure mode
   Sec.~III-C.5 already documents in the aggregate.

Both instances are found by exhaustively scoring every DeepAccident
instance that has a native label (train+val+test, matching the population
`v1_full_eval.py`/`gssm_baseline_eval.py` already score) with the SAME
cached `PreprocessedInstance` fields (`tau_3d`, `tau_phy`,
`evasive_severity`) those scripts use, then filtering/ranking by the
criteria above -- see `search_candidates()`. The exact selected instance
ids and the numbers that justify each selection are written into the
output JSON's `"selection"` block, not just asserted in this docstring.

Per-timestep scoring protocol (4 methods, T=41 frames, dt=0.1s, 4s window
-- the v1 CARM checkpoint's own `cfg.horizon_s`/`cfg.dt`)
--------------------------------------------------------------------------
For every raw frame t of the selected instance:

1. **Raw 3D-TTC**: worst-case (min-over-agents) `tau_3D(t)` from the
   already-cached `pre.tau_3d[agent]` per-frame arrays, then
   `futurerisk.evaluation.raw_ttc_baseline_score` applied ELEMENTWISE
   across the time axis -- the exact same clip(1 - tau/ttc_cap, 0, 1)
   transform used instance-wide everywhere else in this project, just
   evaluated at every frame instead of once on the instance-level min.
2. **RSS**: for every agent, `futurerisk.baselines_extra.rss_margin`
   (already a per-frame array), worst-case (min-over-agents) at each
   frame, then `futurerisk.baselines_extra.rss_score(..., scale=150.0)`
   elementwise -- the same scale constant `v1_full_eval.py` already uses,
   not re-derived here.
3. **GSSM**: `futurerisk.baselines_gssm.gssm_current_acc_features` per
   (agent, frame) -> one batched call to `score_rows_via_subprocess`
   (isolated `.venv_gssm`) -> per-(agent, frame) intensity, worst-case
   (MAX-over-agents, since higher intensity = riskier) at each frame --
   the per-frame generalization of `instance_gssm_score_from_intensity`'s
   worst-agent/worst-moment convention.
4. **FutureRisk (CARM)** -- EXPANDING-WINDOW re-scoring, a deliberate,
   non-obvious protocol specific to THIS figure (documented here
   explicitly so it is never confused with Table I's R, which is computed
   ONCE from the full window):

   CARM's own R is normally a single instance-level scalar computed once
   from the complete window (`predict_R`, `scripts/
   evaluate_crossdomain_v1_to_bench2drive.py`). To get a genuine,
   non-fabricated per-frame R(t) for this figure specifically, at every
   frame index t we TRUNCATE the instance to frames [0, t] only (never
   looking ahead), re-run `preprocess_instance` (recomputing tau_3D,
   interaction type, evasive severity, tau_phy from scratch on the
   truncated window only -- a real, bounded-horizon recomputation, not a
   slice of the full-window arrays) and `build_instance_batch_item_v1` +
   the v1 CARM checkpoint's forward pass on that truncated window, and
   record the resulting scalar R as CARM's score at time t. This makes
   CARM's score at each t use only information up to t, exactly like the
   other three methods' instantaneous per-frame scores -- a fair,
   apples-to-apples comparison. `np.gradient`-based feature terms
   (speed/acceleration/yaw-rate) need >=2 frames, so the shortest
   evaluable window is frames [0,1]; its R is also assigned to t=0 (no
   shorter window is computable, a boundary artifact, not a real distinct
   t=0 estimate -- flagged in the output JSON via
   `"carm_t0_is_boundary_copy": true`).

Score-scale honesty note (see the render script for how this is actually
drawn): CARM/RSS/raw-3D-TTC are all, by construction, bounded to [0,1]
(sigmoid / clip outputs) and so share one meaningful numeric scale. GSSM's
"intensity" (log10 of an equivalent-draw count, `baselines_gssm.py`) is
NOT naturally bounded to [0,1] -- it is a different unit entirely. No
rescaling is invented here; the raw per-frame series for all 4 methods are
saved as-is, and the rendering script documents its own choice of how it
maps each method's raw values to a color scale.

Outputs
-------
`dataset/results_v1/qualitative_per_timestep_comparison.json`:
  - `selection`: search diagnostics for both scenarios (candidate counts,
    the selected instance's own numbers, and the runner-up).
  - `instances.near_miss` / `instances.missed_collision`: for each,
    instance_id, split, native_collision_label, collision timing (if any),
    per-agent geometric collision flags, the highlighted interacting
    agent id, raw ego/agent box trajectories (for the bird's-eye render),
    and every method's raw per-frame time series.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futurerisk.baselines_extra import rss_margin, rss_score
from futurerisk.baselines_gssm import (
    GSSMUnavailableError, gssm_current_acc_features, score_rows_via_subprocess,
)
from futurerisk.data_pipeline import (
    RawTrajectoryInstance, load_preprocessed_instance, preprocess_instance,
)
from futurerisk.evaluation import raw_ttc_baseline_score
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.geometry import BoxTrajectory
from futurerisk.safety_prior import context_dependent_margin
from scripts.evaluate_crossdomain_v1_to_bench2drive import load_checkpoint, predict_R

DEEPACCIDENT_PREPROCESSED = Path("dataset/DeepAccident-preprocessed")
NATIVE_LABELS_PATH = DEEPACCIDENT_PREPROCESSED / "native_collision_labels.json"
THRESHOLD_TABLE_PATH = DEEPACCIDENT_PREPROCESSED / "DeepAccident_thresholds.json"
RESULTS_DIR = Path("dataset/results_v1")
LOGS_DIR = Path("dataset/logs")


def setup_logging(log_path: Path):
    log = logging.getLogger("qualitative_per_timestep_comparison")
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


# --------------------------------------------------------------------------
# Scenario search (real, exhaustive, over every DeepAccident instance with a
# native label -- train+val+test, same population v1_full_eval.py /
# gssm_baseline_eval.py score).
# --------------------------------------------------------------------------

_PLAUSIBLE_SPEED_CAP_MPS = {"pedestrian": 10.0, "cyclist": 10.0}  # ~36 km/h, generous
_PLAUSIBLE_SPEED_CAP_DEFAULT = 40.0  # ~144 km/h, generous for vehicle/other/unknown


def _instance_stat_row(iid: str, part: str, pre, entry: dict) -> Optional[dict]:
    """Note: whenever ANY agent's tau_3D(t) is finite anywhere in the
    window, `min(tau_3D)` over the whole trajectory is trivially exactly
    0.0 (the frame where the real overlap occurs always has tau_3D==0 at
    delta=0 for itself) -- so `min_tau3d`/`worst_margin` are NOT "how far
    in advance was danger detected" signals, they are "did a real
    geometric box-overlap happen ANYWHERE in this window" signals (and,
    via `context_dependent_margin`, roughly how large that agent's own
    tau_phy safety buffer was). This is intentional and matches how
    `safety_prior.analytic_trajectory_risk`/y_analytic already behaves
    project-wide (phi saturates whenever any real overlap occurs) -- not a
    bug introduced here, just documented explicitly since it is easy to
    misread `worst_margin` as a graded "how close" signal when it is
    closer to a "did it touch, and for how forgiving an agent class"
    signal. `select_near_miss` additionally records which SPECIFIC agent
    achieves the worst margin, and its own class + observed max speed, so
    a plausibility filter can be applied before ranking (see
    `select_near_miss`'s docstring for why this matters)."""
    if pre.ego_box is None or not pre.tau_3d:
        return None
    tau3d_stack = np.stack([pre.tau_3d[a] for a in pre.tau_3d])
    tau3d_worst_t = np.min(tau3d_stack, axis=0)
    raw_ttc_t = raw_ttc_baseline_score(tau3d_worst_t, ttc_cap=5.0)
    margins = {a: context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d}
    margin_worst_t = np.min(np.stack(list(margins.values())), axis=0)
    sev_max = max(float(np.max(pre.evasive_severity[a])) for a in pre.evasive_severity)

    worst_agent = min(margins, key=lambda a: np.min(margins[a]))
    worst_agent_class = (pre.agent_classes or {}).get(worst_agent, "unknown")
    worst_agent_box = pre.agent_boxes[worst_agent] if pre.agent_boxes else None
    if worst_agent_box is not None:
        wa_speed = np.hypot(np.gradient(worst_agent_box.x, 0.1), np.gradient(worst_agent_box.y, 0.1))
        worst_agent_max_speed = float(np.max(wa_speed))
    else:
        worst_agent_max_speed = None

    return dict(
        instance_id=iid, split=part,
        native_collision_label=int(entry["collision_label"]),
        collision_intensity=entry.get("collision_intensity"),
        collision_fix_applied=entry.get("collision_fix_applied"),
        worst_margin=float(np.min(margin_worst_t)),
        max_raw_ttc_score=float(np.max(raw_ttc_t)),
        min_tau3d=float(np.min(tau3d_worst_t)),
        max_evasive_severity=sev_max,
        n_agents=len(pre.tau_3d),
        worst_margin_agent_id=worst_agent,
        worst_margin_agent_class=worst_agent_class,
        worst_margin_agent_max_speed=worst_agent_max_speed,
    )


def search_candidates(split: dict, native_labels: dict, log) -> List[dict]:
    """Exhaustively score every DeepAccident instance with a native label
    from its cached PreprocessedInstance (tau_3d/tau_phy/evasive_severity
    already computed and stored -- no recomputation needed for the
    search itself, only for the expanding-window CARM pass on the two
    FINAL selected instances)."""
    rows = []
    n_files = 0
    for part in ("test", "val", "train"):
        for iid in split[part]:
            p = DEEPACCIDENT_PREPROCESSED / f"{iid}.npz"
            if not p.exists():
                continue
            entry = native_labels.get(iid)
            if entry is None or entry.get("collision_label") is None:
                continue
            pre = load_preprocessed_instance(str(p))
            row = _instance_stat_row(iid, part, pre, entry)
            n_files += 1
            if row is not None:
                rows.append(row)
        log.info(f"      ... {part} split scanned ({n_files} files so far, {len(rows)} usable rows)")
    return rows


def _plausible_worst_agent(r: dict) -> bool:
    """Excludes candidates whose worst-margin agent's own observed max
    speed is not physically plausible for its assigned class (e.g. a
    "pedestrian" moving at 40-56 m/s / 145-200 km/h) -- a real, checkable
    DeepAccident data-quality issue (see `futurerisk/adapters/
    deepaccident.py`'s own documented "-inf sentinel" box-dimension quirk
    and Waymo-type-code caveat), NOT an invented exclusion rule. Manual
    inspection of this search's raw ranked list (every one of the top
    ~20 candidates by |worst_margin| was checked by hand before writing
    this filter) found EVERY implausible-speed case was class="pedestrian"
    with a peak speed of 15-56 m/s -- physically impossible for a person
    on foot -- while every class="other"/"vehicle" case had a sane peak
    speed (8-25 m/s, ordinary urban driving); this filter formalizes
    exactly the distinction manual inspection already drew, so the
    resulting top candidate is not silently different from what a human
    reviewer would have picked by hand."""
    cap = _PLAUSIBLE_SPEED_CAP_MPS.get(r["worst_margin_agent_class"], _PLAUSIBLE_SPEED_CAP_DEFAULT)
    speed = r["worst_margin_agent_max_speed"]
    return speed is not None and speed <= cap


def select_near_miss(rows: List[dict], log) -> dict:
    """Scenario 1: native_collision_label==0 (avoided), a real geometric
    close call (worst_margin < 0, i.e. m(t) went negative at some frame --
    eq:margin's tau_3D(t) dipped below tau_safe(t)), with a finite
    min_tau3d (a genuine projected near-approach). Ranked by most negative
    worst_margin (deepest safe-TTC violation) among these, AFTER excluding
    candidates whose worst-margin agent fails the physical-plausibility
    speed check (`_plausible_worst_agent` -- see its docstring: this
    removes a real, checkable class of DeepAccident data-quality
    artifacts, not an arbitrary tightening of the search)."""
    cands_raw = [r for r in rows if r["native_collision_label"] == 0
                 and np.isfinite(r["worst_margin"]) and r["worst_margin"] < 0
                 and np.isfinite(r["min_tau3d"])]
    n_excluded = sum(1 for r in cands_raw if not _plausible_worst_agent(r))
    cands = [r for r in cands_raw if _plausible_worst_agent(r)]
    cands.sort(key=lambda r: r["worst_margin"])
    log.info(f"[search] Scenario 1 (avoided near-miss) candidates: {len(cands_raw)} raw, "
              f"{n_excluded} excluded by the worst-agent speed-plausibility filter, {len(cands)} remaining")
    for r in cands[:5]:
        log.info(f"         {r['instance_id']:60s} split={r['split']:5s} worst_margin={r['worst_margin']:+.3f}"
                  f" min_tau3d={r['min_tau3d']:.3f} max_evasive_sev={r['max_evasive_severity']:.3f}"
                  f" worst_agent={r['worst_margin_agent_id']}({r['worst_margin_agent_class']},"
                  f" {r['worst_margin_agent_max_speed']:.1f}m/s) n_agents={r['n_agents']}")
    if not cands:
        raise RuntimeError("No plausible collision-free instance with a negative worst_margin found -- cannot proceed.")
    return {"selected": cands[0], "runner_up": cands[1] if len(cands) > 1 else None,
            "n_candidates": len(cands), "n_candidates_before_plausibility_filter": len(cands_raw),
            "n_excluded_by_plausibility_filter": n_excluded}


def select_missed_collision(rows: List[dict], log) -> dict:
    """Scenario 2: native_collision_label==1 (a real, native-anchored
    collision), ranked ascending by max_raw_ttc_score (the per-instance
    max over time of the elementwise raw-3D-TTC score) -- i.e. the
    instance where the worst-case tau_3D(t) came CLOSEST to never
    dropping below the forward-simulation cap anywhere in the window."""
    cands = [r for r in rows if r["native_collision_label"] == 1]
    cands.sort(key=lambda r: r["max_raw_ttc_score"])
    n_zero = sum(1 for r in cands if r["max_raw_ttc_score"] < 1e-9)
    log.info(f"[search] Scenario 2 (native collision) candidates: {len(cands)}"
              f"; with max_raw_ttc_score==0.0 exactly: {n_zero}")
    for r in cands[:5]:
        log.info(f"         {r['instance_id']:60s} split={r['split']:5s} max_raw_ttc_score={r['max_raw_ttc_score']:.4f}"
                  f" min_tau3d={r['min_tau3d']:.3f} collision_intensity={r['collision_intensity']}"
                  f" collision_fix_applied={r['collision_fix_applied']} n_agents={r['n_agents']}")
    if not cands:
        raise RuntimeError("No native-collision instance found -- cannot proceed.")
    return {"selected": cands[0], "runner_up": cands[1] if len(cands) > 1 else None,
            "n_candidates": len(cands), "n_with_zero_raw_ttc_score": n_zero}


# --------------------------------------------------------------------------
# Per-timestep scoring, 4 methods
# --------------------------------------------------------------------------

def _pick_interacting_agent(pre) -> str:
    """The single agent drawn in each panel: the one this instance's own
    "close call" story is actually about -- the agent achieving the worst
    (most negative) closed-form margin m^j(t) = tau_3D^j(t) - tau_phy^j(t)
    over the whole window (eq:margin), the same quantity `select_near_miss`
    ranks candidate instances by. Falls back to worst raw tau_3D (the
    `v1_full_eval.instance_dominant_type` tie-breaking convention) if no
    agent has a finite margin anywhere (e.g. every agent's tau_3D stayed
    at +infinity the whole window, the missed-collision scenario's own
    defining property -- there `build_instance_entry`'s `highlight_agent`
    override, which prefers whichever agent's native-anchored collision
    flag actually fired, takes precedence over this function's choice
    anyway)."""
    margins = {a: context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d}
    finite_agents = [a for a in margins if np.any(np.isfinite(margins[a]))]
    if finite_agents:
        return min(finite_agents, key=lambda a: np.min(margins[a]))
    return min(pre.tau_3d, key=lambda a: np.min(pre.tau_3d[a]))


def score_raw_ttc(pre, cfg) -> np.ndarray:
    tau3d_worst_t = np.min(np.stack([pre.tau_3d[a] for a in pre.tau_3d]), axis=0)
    return raw_ttc_baseline_score(tau3d_worst_t, ttc_cap=cfg.ttc_cap)


def score_rss(raw: RawTrajectoryInstance, pre, cfg) -> np.ndarray:
    margins = []
    for aid, agent_box in raw.agent_boxes.items():
        itype = pre.interaction_type.get(aid, np.zeros(len(agent_box), dtype=int))
        margins.append(rss_margin(raw.ego_box, agent_box, cfg.dt, itype))
    margin_worst_t = np.min(np.stack(margins), axis=0)
    return rss_score(margin_worst_t, scale=150.0)


def score_gssm(raw: RawTrajectoryInstance, cfg, batch_tag: str) -> np.ndarray:
    """Per-frame worst-case (MAX-over-agents) GSSM intensity -- one batched
    subprocess call across every agent of this single instance."""
    feats, spacings, lens = [], [], []
    for agent_box in raw.agent_boxes.values():
        f, s = gssm_current_acc_features(raw.ego_box, agent_box, cfg.dt)
        feats.append(f)
        spacings.append(s)
        lens.append(len(s))
    X = np.concatenate(feats, axis=0)
    s = np.concatenate(spacings, axis=0)
    intensity = score_rows_via_subprocess(X, s, batch_tag=batch_tag)
    T = lens[0]
    per_agent = intensity.reshape(len(lens), T)
    return np.max(per_agent, axis=0)


def _truncate_box(box: BoxTrajectory, n: int) -> BoxTrajectory:
    return BoxTrajectory(x=box.x[:n], y=box.y[:n], z=box.z[:n], heading=box.heading[:n],
                          length=box.length[:n], width=box.width[:n], height=box.height[:n])


def _truncate_raw_instance(raw: RawTrajectoryInstance, n: int) -> RawTrajectoryInstance:
    return RawTrajectoryInstance(
        dataset=raw.dataset, instance_id=f"{raw.instance_id}__trunc{n:02d}",
        times=raw.times[:n], ego_box=_truncate_box(raw.ego_box, n),
        agent_boxes={aid: _truncate_box(b, n) for aid, b in raw.agent_boxes.items()},
        agent_collision_flags={aid: f[:n] for aid, f in raw.agent_collision_flags.items()},
        lane_graph_available=False,
        agent_classes=raw.agent_classes,
    )


def score_carm_expanding_window(raw: RawTrajectoryInstance, cfg, threshold_table, model, log) -> np.ndarray:
    """CARM per-frame R(t) via expanding-window re-scoring -- see module
    docstring for the full protocol and why this differs from CARM's
    normal single-scalar, full-window R used everywhere else in this
    project. Never looks ahead of frame t."""
    T = len(raw.times)
    R_t = np.zeros(T, dtype=float)
    for t in range(T):
        n = min(max(t + 1, 2), T)  # >=2 frames needed for np.gradient-based features
        truncated = _truncate_raw_instance(raw, n)
        pre_t = preprocess_instance(truncated, threshold_table, cfg=cfg)
        R_t[t] = predict_R(model, cfg, truncated, pre_t)
    log.info(f"      CARM expanding-window: {T} frames scored "
              f"(t=0 borrows the n=2 window's R, a boundary artifact -- see docstring)")
    return R_t


def collision_frame_index(raw: RawTrajectoryInstance, interacting_agent: str) -> Optional[int]:
    flags = raw.agent_collision_flags.get(interacting_agent)
    if flags is None or not flags.any():
        # fall back to ANY agent's flag (the collision partner might not be
        # the min-tau3d agent for the missed-collision scenario, since
        # tau_3D never triggers for it by construction)
        for aid, f in raw.agent_collision_flags.items():
            if f.any():
                return int(np.argmax(f))
        return None
    return int(np.argmax(flags))


def build_instance_entry(
    iid: str, split_name: str, model, cfg, threshold_table, log, gssm_tag: str,
    prefer_native_collision_partner: bool,
) -> dict:
    p = DEEPACCIDENT_PREPROCESSED / f"{iid}.npz"
    pre = load_preprocessed_instance(str(p))
    raw = RawTrajectoryInstance(
        dataset=pre.dataset, instance_id=pre.instance_id, times=pre.times,
        ego_box=pre.ego_box, agent_boxes=pre.agent_boxes,
        agent_collision_flags=pre.agent_collision_flags, lane_graph_available=False,
        agent_classes=pre.agent_classes,
    )
    interacting_agent = _pick_interacting_agent(pre)
    # For the missed-collision scenario ONLY: the actual (native-anchored)
    # collision partner is, by construction, an agent whose own tau_3D
    # never triggers, so `interacting_agent` (chosen by worst margin/tau_3D)
    # would NOT point at it -- prefer whichever agent's flag actually fired.
    # For the near-miss scenario this override is NOT applied: a dense
    # DeepAccident scene often has >1 tracked agent (of dozens) with SOME
    # incidental geometric flag, and we specifically want the ONE agent
    # that drove the worst-margin selection, not an arbitrary other one.
    highlight_agent = interacting_agent
    if prefer_native_collision_partner:
        collided_agents = [aid for aid, f in raw.agent_collision_flags.items() if f.any()]
        if collided_agents:
            highlight_agent = collided_agents[0]

    log.info(f"      scoring raw-3D-TTC ...")
    raw_ttc_t = score_raw_ttc(pre, cfg)
    log.info(f"      scoring RSS ...")
    rss_t = score_rss(raw, pre, cfg)
    log.info(f"      scoring GSSM (subprocess) ...")
    gssm_t = score_gssm(raw, cfg, batch_tag=gssm_tag)
    log.info(f"      scoring CARM (expanding-window, {len(raw.times)} truncated re-scores) ...")
    t_carm0 = time.time()
    carm_t = score_carm_expanding_window(raw, cfg, threshold_table, model, log)
    log.info(f"      CARM expanding-window done in {time.time()-t_carm0:.1f}s")

    coll_idx = collision_frame_index(raw, highlight_agent)

    # Worst-case (min-over-agents) closed-form margin m(t) (eq:margin), for
    # marking the near-miss column's "closest approach" frame in the
    # render script -- the same quantity `select_near_miss` ranked
    # candidates by, just kept per-frame here instead of collapsed to one
    # instance-level scalar.
    margins = [context_dependent_margin(pre.tau_3d[a], pre.tau_phy[a]) for a in pre.tau_3d]
    margin_t = np.min(np.stack(margins), axis=0)
    closest_idx = int(np.argmin(margin_t))

    return dict(
        instance_id=iid, split=split_name, times=pre.times.tolist(),
        interacting_agent_id=interacting_agent, highlight_agent_id=highlight_agent,
        n_agents=len(pre.agent_boxes),
        native_collision_label=None,  # filled by caller
        # collision_frame_index/collision_time_s reflect `highlight_agent`'s
        # own geometric SAT collision flag (`agent_collision_flags`), which
        # is NOT the same thing as a native-confirmed collision -- for the
        # near-miss instance (native_collision_label will be set to 0 by
        # the caller) this is a real geometric box-touch that native ground
        # truth did NOT count as an actual accident; only read this as "the
        # actual collision" when native_collision_label==1 (see
        # `render_qualitative_per_timestep_comparison.py`, which gates its
        # "actual collision" marker on exactly that).
        collision_frame_index=coll_idx,
        collision_time_s=float(pre.times[coll_idx]) if coll_idx is not None else None,
        margin_t=margin_t.tolist(),
        closest_approach_frame_index=closest_idx,
        closest_approach_margin=float(margin_t[closest_idx]),
        ego_box=dict(x=raw.ego_box.x.tolist(), y=raw.ego_box.y.tolist(), heading=raw.ego_box.heading.tolist(),
                     length=raw.ego_box.length.tolist(), width=raw.ego_box.width.tolist()),
        agent_box=dict(x=raw.agent_boxes[highlight_agent].x.tolist(), y=raw.agent_boxes[highlight_agent].y.tolist(),
                       heading=raw.agent_boxes[highlight_agent].heading.tolist(),
                       length=raw.agent_boxes[highlight_agent].length.tolist(),
                       width=raw.agent_boxes[highlight_agent].width.tolist()),
        scores=dict(
            futurerisk_carm=carm_t.tolist(),
            rss=rss_t.tolist(),
            raw_ttc=raw_ttc_t.tolist(),
            gssm=gssm_t.tolist(),
        ),
        carm_t0_is_boundary_copy=True,
    )


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(LOGS_DIR / f"qualitative_per_timestep_comparison_{datetime.now():%Y%m%d_%H%M%S}.log")
    t0 = time.time()

    log.info("[1/5] Loading v1 CARM checkpoint + DeepAccident threshold table + native labels ...")
    model, cfg, ckpt = load_checkpoint()
    split = ckpt["split"]
    threshold_table = load_threshold_tables(str(THRESHOLD_TABLE_PATH))["DeepAccident"]
    with open(NATIVE_LABELS_PATH) as f:
        native_labels = json.load(f)
    log.info(f"      cfg: dt={cfg.dt} horizon_s={cfg.horizon_s} num_bins={cfg.num_bins} ttc_cap={cfg.ttc_cap}")

    log.info("[2/5] Searching every DeepAccident instance with a native label for both scenarios ...")
    rows = search_candidates(split, native_labels, log)
    log.info(f"      total usable instances scanned: {len(rows)}")

    near_miss_sel = select_near_miss(rows, log)
    missed_sel = select_missed_collision(rows, log)

    near_miss_iid = near_miss_sel["selected"]["instance_id"]
    near_miss_split = near_miss_sel["selected"]["split"]
    missed_iid = missed_sel["selected"]["instance_id"]
    missed_split = missed_sel["selected"]["split"]
    log.info(f"[3/5] Selected near-miss instance: {near_miss_iid} (split={near_miss_split})")
    log.info(f"      Selected missed-collision instance: {missed_iid} (split={missed_split})")

    log.info("[4/5] Scoring all 4 methods per-frame for the near-miss instance ...")
    try:
        near_miss_entry = build_instance_entry(
            near_miss_iid, near_miss_split, model, cfg, threshold_table, log, gssm_tag="qual_near_miss",
            prefer_native_collision_partner=False,
        )
    except GSSMUnavailableError as exc:
        log.error(f"GSSM scoring failed (see futurerisk/baselines_gssm.py's isolated-venv setup): {exc}")
        sys.exit(1)
    near_miss_entry["native_collision_label"] = 0
    near_miss_entry["selection_stats"] = near_miss_sel["selected"]

    log.info("[5/5] Scoring all 4 methods per-frame for the missed-collision instance ...")
    try:
        missed_entry = build_instance_entry(
            missed_iid, missed_split, model, cfg, threshold_table, log, gssm_tag="qual_missed_collision",
            prefer_native_collision_partner=True,
        )
    except GSSMUnavailableError as exc:
        log.error(f"GSSM scoring failed (see futurerisk/baselines_gssm.py's isolated-venv setup): {exc}")
        sys.exit(1)
    missed_entry["native_collision_label"] = 1
    missed_entry["selection_stats"] = missed_sel["selected"]

    results = {
        "protocol_notes": {
            "T": len(near_miss_entry["times"]),
            "dt": cfg.dt,
            "horizon_s": cfg.horizon_s,
            "carm_protocol": "expanding-window re-scoring per frame, see module docstring; "
                              "NOT the same as CARM's normal single-scalar full-window R used in Table I.",
            "raw_ttc_transform": "futurerisk.evaluation.raw_ttc_baseline_score, elementwise over time",
            "rss_scale": 150.0,
            "gssm_aggregation": "max over agents per frame (higher intensity = riskier)",
        },
        "selection": {
            "near_miss": {k: v for k, v in near_miss_sel.items()},
            "missed_collision": {k: v for k, v in missed_sel.items()},
        },
        "instances": {
            "near_miss": near_miss_entry,
            "missed_collision": missed_entry,
        },
    }

    out_path = RESULTS_DIR / "qualitative_per_timestep_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"wrote {out_path}")
    log.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
