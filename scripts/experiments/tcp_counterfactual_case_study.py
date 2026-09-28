#!/usr/bin/env python3
"""Open-loop counterfactual case study for TCP (mirrors
admlp_counterfactual_case_study.py): what would FutureRisk (v1) have scored
if TCP's predicted trajectory -- conditioned on the real front-camera image,
not just ego kinematics like AD-MLP -- had been driven, in the same real
Bench2Drive scenario?

Unlike the AD-MLP script (which only checked each clip's earliest valid
reference frame), this scans several reference frames per clip and keeps
the one with the largest TCP prediction error, since TCP is accurate enough
on nominal driving (mean L2 ~0.3-0.6m over 2s at frame 0, see
tcp_predictions.json) that an interesting divergence is more likely deeper
into a scenario's hazard than at its start.

TCP only predicts 2.0s (4 waypoints at 0.5s spacing); the remaining 2.0s of
the v1 checkpoint's fixed 4s horizon is filled by constant-velocity
extrapolation from the final predicted waypoint, exactly as
admlp_counterfactual_case_study.py extrapolates AD-MLP's last 1s -- clearly
marked as extrapolated, not fabricated real data.
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
from futurerisk.data_pipeline import preprocess_instance
from futurerisk.data_pipeline import RawTrajectoryInstance
from futurerisk.event_proxy import load_threshold_tables
from futurerisk.geometry import BoxTrajectory, boxes_overlap, heading_from_positions
from scripts.experiments.tcp_case_study import build_tcp_input, load_tcp_model, predict_future_global
from scripts.evaluate_crossdomain_v1_to_bench2drive import DEVICE, load_checkpoint, predict_R

WINDOW_S = 4.0  # matches the v1 checkpoint's fixed horizon
DT = 0.1
WINDOW_FRAMES = int(round(WINDOW_S / DT)) + 1  # 41
TCP_HORIZON_S = 2.0  # TCP predicts 4 waypoints at 0.5s spacing


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

    # counterfactual ego: real position at t=0, then TCP's 4 predicted
    # waypoints (0.5s apart) linearly interpolated onto the dt=0.1 grid,
    # extended for the remaining 2s by constant-velocity extrapolation from
    # the final predicted waypoint (clearly an extrapolation, not real data).
    t_known = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    xy_known = np.concatenate([[[ego_real.x[0], ego_real.y[0]]], xy_pred], axis=0)
    t_grid = np.arange(WINDOW_FRAMES) * DT
    x_interp = np.interp(np.clip(t_grid, 0, TCP_HORIZON_S), t_known, xy_known[:, 0])
    y_interp = np.interp(np.clip(t_grid, 0, TCP_HORIZON_S), t_known, xy_known[:, 1])
    beyond = t_grid > TCP_HORIZON_S
    if beyond.any():
        vx = (xy_known[-1, 0] - xy_known[-2, 0]) / 0.5
        vy = (xy_known[-1, 1] - xy_known[-2, 1]) / 0.5
        extra_t = t_grid[beyond] - TCP_HORIZON_S
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
        dataset="Bench2Drive", instance_id=f"{clip_name}__r{ref_frame:05d}__tcp_cf",
        times=times, ego_box=ego_cf, agent_collision_flags=collision_flags_cf, **common,
    )
    return real_inst, cf_inst


_THRESHOLD_TABLE = load_threshold_tables("dataset/Bench2Drive-Base-preprocessed/Bench2Drive_thresholds.json")["Bench2Drive"]


def score_instance(model, cfg, inst) -> dict:
    pre = preprocess_instance(inst, _THRESHOLD_TABLE, cfg=cfg)
    R = predict_R(model, cfg, inst, pre)
    geo_collision = int(any(f.any() for f in inst.agent_collision_flags.values()))
    return {"R": R, "geometric_collision": geo_collision, "n_agents": len(inst.agent_boxes)}


def candidate_reference_frames(frames, stride=20):
    """Scan every `stride`-th frame that has a full 2s future window,
    skipping the clip's very first few frames (degenerate startup state)."""
    idxs = sorted(frames.keys())
    out = []
    for i in idxs[10::stride]:
        if i + 20 <= idxs[-1] and all((i + d) in frames for d in range(0, 21)):
            out.append(i)
    return out


def main():
    tcp_net = load_tcp_model()
    carm_model, carm_cfg, ckpt = load_checkpoint()

    clip_files = sorted(Path("dataset/Bench2Drive-Base").glob("*.tar.gz"))
    candidates = [f for f in clip_files if any(k in f.name for k in ("Accident", "Crossing", "Junction", "Cut"))]
    print(f"{len(candidates)} candidate clips")

    results = []
    for f in candidates[:15]:
        clip_name, frames = load_clip_frames(str(f))
        ref_frames = candidate_reference_frames(frames)
        if not ref_frames:
            continue
        # pick the reference frame with the largest TCP prediction error
        best = None
        for i in ref_frames:
            xy_pred, _ = predict_future_global(tcp_net, frames, str(f), i)
            real_future = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20]])
            l2 = np.linalg.norm(xy_pred - real_future, axis=1).mean()
            if best is None or l2 > best[2]:
                best = (i, xy_pred, l2)
        i, xy_pred, l2_mean = best
        try:
            real_inst, cf_inst = build_real_and_counterfactual(str(f), i, xy_pred, DEFAULT_CONFIG)
        except ValueError as e:
            print(f"  {clip_name}: skip ({e})")
            continue
        real_score = score_instance(carm_model, carm_cfg, real_inst)
        cf_score = score_instance(carm_model, carm_cfg, cf_inst)
        row = {
            "clip_name": clip_name, "ref_frame": i, "tcp_l2_mean": float(l2_mean),
            "n_agents": real_score["n_agents"],
            "R_real": real_score["R"], "collision_real": real_score["geometric_collision"],
            "R_tcp_counterfactual": cf_score["R"], "collision_tcp_counterfactual": cf_score["geometric_collision"],
        }
        results.append(row)
        print(f"  {clip_name}  ref={i}  n_agents={row['n_agents']}  tcp_L2={row['tcp_l2_mean']:.2f}m"
              f"  R_real={row['R_real']:.3f}(coll={row['collision_real']})"
              f"  R_tcp_cf={row['R_tcp_counterfactual']:.3f}(coll={row['collision_tcp_counterfactual']})")

    Path("dataset/results_v1").mkdir(parents=True, exist_ok=True)
    with open("dataset/results_v1/tcp_counterfactual_case_studies.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote dataset/results_v1/tcp_counterfactual_case_studies.json ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
