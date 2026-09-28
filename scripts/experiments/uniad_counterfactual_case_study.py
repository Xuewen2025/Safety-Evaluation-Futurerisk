#!/usr/bin/env python3
"""Open-loop counterfactual case study for UniAD-Tiny (mirrors
tcp_counterfactual_case_study.py/admlp_counterfactual_case_study.py): what
would FutureRisk (v1) have scored if UniAD's predicted trajectory -- a full
surround-view-camera + BEV planner, unlike TCP's single-front-camera model
or AD-MLP's kinematics-only model -- had been driven, in the same real
Bench2Drive scenario?

ENVIRONMENT: unlike stage A (uniad_case_study.py, which must run inside
external/Bench2DriveZoo/.venv_uniad), this script runs in the MAIN system
Python (the one with `futurerisk` and the CARM v1 checkpoint) -- it only
reads dataset/results_v1/uniad_predictions.json, produced by stage A ahead
of time, and never imports mmcv/torch-cu-extension code. Run it exactly
like the TCP/AD-MLP counterfactual scripts:

    python3 scripts/experiments/uniad_counterfactual_case_study.py

Prerequisite: run `external/Bench2DriveZoo/.venv_uniad/bin/python
scripts/experiments/uniad_case_study.py` first (or a version of it with
more/different candidate clips) to (re)generate
dataset/results_v1/uniad_predictions.json.

UniAD's planning head has a fixed 3.0s / 6-waypoint output (0.5s spacing),
shorter than the v1 CARM checkpoint's fixed 4.0s hazard-head horizon, so
(exactly as tcp_counterfactual_case_study.py does for TCP's 2.0s and
admlp_counterfactual_case_study.py does for AD-MLP's 3.0s) the remaining
1.0s is filled by constant-velocity extrapolation from the final predicted
waypoint -- clearly marked as extrapolated, not fabricated real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.adapters.bench2drive import (
    _build_agent_track, _collect_agent_bboxes_by_position, _extract_ego_track, _lane_change_event_times,
    _longest_contiguous_run, _slice_box, load_clip_frames,
)
from futurerisk.config import DEFAULT_CONFIG, FutureRiskConfig
from futurerisk.data_pipeline import RawTrajectoryInstance, preprocess_instance
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.geometry import BoxTrajectory, boxes_overlap, heading_from_positions
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint, predict_R

WINDOW_S = 4.0  # matches the v1 checkpoint's fixed horizon
DT = 0.1
WINDOW_FRAMES = int(round(WINDOW_S / DT)) + 1  # 41
UNIAD_HORIZON_S = 3.0  # UniAD's planning head predicts 6 waypoints at 0.5s spacing
PREDICTIONS_PATH = Path("dataset/results_v1/uniad_predictions.json")
CF_SUFFIX = "uniad_cf"


def build_real_and_counterfactual(tar_path: str, ref_frame: int, xy_pred: np.ndarray, cfg: FutureRiskConfig):
    """Identical in structure to tcp_counterfactual_case_study.py's function
    of the same name, generalized from 4 to 6 known future waypoints (0.5s
    apart) and from a 2.0s to a 3.0s known horizon before falling back to
    constant-velocity extrapolation."""
    clip_name, frames = load_clip_frames(tar_path)
    all_idxs = sorted(frames.keys())
    start, end = _longest_contiguous_run(all_idxs)
    idxs = all_idxs[start:end]
    pos_i = idxs.index(ref_frame)
    if pos_i + WINDOW_FRAMES > len(idxs):
        raise ValueError("not enough frames after reference for a full window")

    ego_box_full, ego_road_lane = _extract_ego_track(frames, idxs, cfg)
    ego_real = _slice_box(ego_box_full, pos_i, pos_i + WINDOW_FRAMES)

    bboxes_by_agent = _collect_agent_bboxes_by_position(frames, idxs)
    agent_boxes, agent_classes = {}, {}
    for agent_id, bbox_by_pos in bboxes_by_agent.items():
        track = _build_agent_track(bbox_by_pos, cfg, min_track_frames=10)
        if track is None:
            continue
        if track.run_start <= pos_i and track.run_end >= pos_i + WINDOW_FRAMES:
            local_start, local_end = pos_i - track.run_start, pos_i + WINDOW_FRAMES - track.run_start
            agent_boxes[agent_id] = _slice_box(track.box, local_start, local_end)
            agent_classes[agent_id] = track.agent_class

    idx_id = np.arange(WINDOW_FRAMES)
    collision_flags_real = {
        aid: boxes_overlap(ego_real, idx_id, box, idx_id, check_vertical=True) for aid, box in agent_boxes.items()
    }

    # counterfactual ego: real position at t=0, then UniAD's 6 predicted
    # waypoints (0.5s apart) linearly interpolated onto the dt=0.1 grid,
    # extended for the remaining 1s by constant-velocity extrapolation from
    # the final predicted waypoint (clearly an extrapolation, not real data).
    t_known = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    xy_known = np.concatenate([[[ego_real.x[0], ego_real.y[0]]], xy_pred], axis=0)
    t_grid = np.arange(WINDOW_FRAMES) * DT
    x_interp = np.interp(np.clip(t_grid, 0, UNIAD_HORIZON_S), t_known, xy_known[:, 0])
    y_interp = np.interp(np.clip(t_grid, 0, UNIAD_HORIZON_S), t_known, xy_known[:, 1])
    beyond = t_grid > UNIAD_HORIZON_S
    if beyond.any():
        vx = (xy_known[-1, 0] - xy_known[-2, 0]) / 0.5
        vy = (xy_known[-1, 1] - xy_known[-2, 1]) / 0.5
        extra_t = t_grid[beyond] - UNIAD_HORIZON_S
        x_interp[beyond] = xy_known[-1, 0] + vx * extra_t
        y_interp[beyond] = xy_known[-1, 1] + vy * extra_t
    z_cf = np.full(WINDOW_FRAMES, ego_real.z[0])
    heading_cf = heading_from_positions(x_interp, y_interp, dt=DT, fallback_heading=ego_real.heading[0])
    ego_cf = BoxTrajectory(
        x=x_interp, y=y_interp, z=z_cf, heading=heading_cf,
        length=ego_real.length, width=ego_real.width, height=ego_real.height,
    )
    collision_flags_cf = {
        aid: boxes_overlap(ego_cf, idx_id, box, idx_id, check_vertical=True) for aid, box in agent_boxes.items()
    }

    times = np.arange(WINDOW_FRAMES) * DT
    ego_lane_change = _lane_change_event_times(ego_road_lane[pos_i:pos_i + WINDOW_FRAMES], times)
    common = dict(
        agent_boxes=agent_boxes, agent_classes=agent_classes,
        lane_graph_available=True, ego_lane_change_times=ego_lane_change,
        agent_lane_change_times={aid: np.array([]) for aid in agent_boxes},
    )
    real_inst = RawTrajectoryInstance(
        dataset="Bench2Drive", instance_id=f"{clip_name}__r{ref_frame:05d}__real",
        times=times, ego_box=ego_real, agent_collision_flags=collision_flags_real, **common,
    )
    cf_inst = RawTrajectoryInstance(
        dataset="Bench2Drive", instance_id=f"{clip_name}__r{ref_frame:05d}__{CF_SUFFIX}",
        times=times, ego_box=ego_cf, agent_collision_flags=collision_flags_cf, **common,
    )
    return real_inst, cf_inst


_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]


def score_instance(model, cfg, inst) -> dict:
    pre = preprocess_instance(inst, _THRESHOLD_TABLE, cfg=cfg)
    R = predict_R(model, cfg, inst, pre)
    geo_collision = int(any(f.any() for f in inst.agent_collision_flags.values()))
    return {"R": R, "geometric_collision": geo_collision, "n_agents": len(inst.agent_boxes)}


def main():
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"{PREDICTIONS_PATH} not found -- run stage A first:\n"
            f"  external/Bench2DriveZoo/.venv_uniad/bin/python scripts/experiments/uniad_case_study.py"
        )
    with open(PREDICTIONS_PATH) as fh:
        predictions = json.load(fh)
    print(f"loaded {len(predictions)} UniAD predictions from {PREDICTIONS_PATH}")

    carm_model, carm_cfg, ckpt = load_checkpoint()

    results = []
    for row in predictions:
        f, i = row["tar_path"], row["ref_frame"]
        xy_pred = np.array(row["pred_xy"])
        l2_mean = float(np.mean(row["l2_per_waypoint"]))
        try:
            real_inst, cf_inst = build_real_and_counterfactual(f, i, xy_pred, DEFAULT_CONFIG)
        except ValueError as e:
            print(f"  {row['clip_name']}: skip ({e})")
            continue
        real_score = score_instance(carm_model, carm_cfg, real_inst)
        cf_score = score_instance(carm_model, carm_cfg, cf_inst)
        out_row = {
            "clip_name": row["clip_name"], "ref_frame": i, "uniad_l2_mean": l2_mean,
            "n_agents": real_score["n_agents"],
            "R_real": real_score["R"], "collision_real": real_score["geometric_collision"],
            "R_uniad_counterfactual": cf_score["R"], "collision_uniad_counterfactual": cf_score["geometric_collision"],
        }
        results.append(out_row)
        print(f"  {out_row['clip_name']}  ref={i}  n_agents={out_row['n_agents']}  uniad_L2={l2_mean:.2f}m"
              f"  R_real={out_row['R_real']:.3f}(coll={out_row['collision_real']})"
              f"  R_uniad_cf={out_row['R_uniad_counterfactual']:.3f}(coll={out_row['collision_uniad_counterfactual']})")

    out_path = Path("dataset/results_v1/uniad_counterfactual_case_studies.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out_path} ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
