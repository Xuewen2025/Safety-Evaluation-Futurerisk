#!/usr/bin/env python3
"""Open-loop counterfactual case study: what would FutureRisk (v1) have
scored if AD-MLP's predicted trajectory (not the recorded expert path) had
been driven, in the same real Bench2Drive scenario?

Construction: real ego box + real agent boxes for the 3s window following a
chosen reference frame (matching how build_instances_from_clip already
slices one clip's tracks), duplicated into two RawTrajectoryInstances that
differ ONLY in ego's box -- one keeps the real recorded ego path, the other
substitutes AD-MLP's predicted path (linearly interpolated between its 6
waypoints at 0.5s spacing, heading re-derived from the interpolated path the
same way every other adapter in this package does it). Every other agent's
track, and its collision flags against each ego variant, are computed
identically otherwise; only the counterfactual's ego-agent collision flags
actually change (since ego's box is different).

This is open-loop: other agents do not react to which ego trajectory is
used (no live simulator), which is exactly what "open-loop, no CARLA"
means here -- documented as a real, disclosed limitation of this
comparison, not presented as equivalent to a closed-loop rollout.

The v1 checkpoint's hazard head has a fixed 20-bin, 4s output -- AD-MLP only
predicts 3s (6 waypoints), so the last 1s (10 frames) is filled by
constant-velocity extrapolation from the final predicted waypoint,
clearly marked as extrapolated, not fabricated real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

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
from scripts.experiments.admlp_case_study import (
    build_admlp_input, find_valid_reference_frame, load_admlp_model, predict_future_global,
)
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint, predict_R

WINDOW_S = 4.0  # matches the v1 checkpoint's fixed horizon
DT = 0.1
WINDOW_FRAMES = int(round(WINDOW_S / DT)) + 1  # 41


def build_real_and_counterfactual(tar_path: str, ref_frame: int, xy_pred: np.ndarray, cfg: FutureRiskConfig):
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

    # counterfactual ego: real position at t=0, then AD-MLP's 6 predicted
    # waypoints (0.5s apart) linearly interpolated onto the dt=0.1 grid,
    # extended for the last 1s by constant-velocity extrapolation from the
    # final predicted waypoint (clearly an extrapolation, not real data).
    t_known = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    xy_known = np.concatenate([[[ego_real.x[0], ego_real.y[0]]], xy_pred], axis=0)
    t_grid = np.arange(WINDOW_FRAMES) * DT
    x_interp = np.interp(np.clip(t_grid, 0, 3.0), t_known, xy_known[:, 0])
    y_interp = np.interp(np.clip(t_grid, 0, 3.0), t_known, xy_known[:, 1])
    beyond = t_grid > 3.0
    if beyond.any():
        vx = (xy_known[-1, 0] - xy_known[-2, 0]) / 0.5
        vy = (xy_known[-1, 1] - xy_known[-2, 1]) / 0.5
        extra_t = t_grid[beyond] - 3.0
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
        dataset="Bench2Drive", instance_id=f"{clip_name}__r{ref_frame:05d}__admlp_cf",
        times=times, ego_box=ego_cf, agent_collision_flags=collision_flags_cf, **common,
    )
    return real_inst, cf_inst


_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]


def score_instance(model, cfg, inst) -> dict:
    # Reuse Bench2Drive-Base's already-fit threshold table (S3.5.2: "a fixed
    # preprocessing artifact ... reused for all subsequent label generation")
    # rather than re-fitting on a single ad-hoc instance, which would be a
    # statistically weak, noisy estimate.
    pre = preprocess_instance(inst, _THRESHOLD_TABLE, cfg=cfg)
    R = predict_R(model, cfg, inst, pre)
    geo_collision = int(any(f.any() for f in inst.agent_collision_flags.values()))
    return {"R": R, "geometric_collision": geo_collision, "n_agents": len(inst.agent_boxes)}


def main():
    admlp_net = load_admlp_model()
    carm_model, carm_cfg, ckpt = load_checkpoint()

    clip_files = sorted(Path("dataset/Bench2Drive-Base").glob("*.tar.gz"))
    candidates = [f for f in clip_files if any(k in f.name for k in ("Accident", "Crossing", "Junction", "Cut"))]
    print(f"{len(candidates)} candidate clips")

    results = []
    for f in candidates[:15]:
        clip_name, frames = load_clip_frames(str(f))
        try:
            i = find_valid_reference_frame(frames)
        except ValueError:
            continue
        xy_pred, _ = predict_future_global(admlp_net, frames, i)
        real_future = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20, 25, 30]])
        l2 = np.linalg.norm(xy_pred - real_future, axis=1)
        try:
            real_inst, cf_inst = build_real_and_counterfactual(str(f), i, xy_pred, DEFAULT_CONFIG)
        except ValueError as e:
            print(f"  {clip_name}: skip ({e})")
            continue
        real_score = score_instance(carm_model, carm_cfg, real_inst)
        cf_score = score_instance(carm_model, carm_cfg, cf_inst)
        row = {
            "clip_name": clip_name, "ref_frame": i, "admlp_l2_mean": float(l2.mean()),
            "n_agents": real_score["n_agents"],
            "R_real": real_score["R"], "collision_real": real_score["geometric_collision"],
            "R_admlp_counterfactual": cf_score["R"], "collision_admlp_counterfactual": cf_score["geometric_collision"],
        }
        results.append(row)
        print(f"  {clip_name}  n_agents={row['n_agents']}  admlp_L2={row['admlp_l2_mean']:.1f}m"
              f"  R_real={row['R_real']:.3f}(coll={row['collision_real']})"
              f"  R_admlp_cf={row['R_admlp_counterfactual']:.3f}(coll={row['collision_admlp_counterfactual']})")

    Path("dataset/results_v1").mkdir(parents=True, exist_ok=True)
    with open("dataset/results_v1/admlp_counterfactual_case_studies.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote dataset/results_v1/admlp_counterfactual_case_studies.json ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
