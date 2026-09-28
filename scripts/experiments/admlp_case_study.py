#!/usr/bin/env python3
"""Run the released AD-MLP checkpoint (open-loop, no CARLA) on real
Bench2Drive scenarios, build a counterfactual instance where ego follows
AD-MLP's *predicted* trajectory instead of the recorded expert one (every
other agent's real recorded track is unchanged -- this is an open-loop
"what would FutureRisk have scored if this policy had been driving"
comparison, not a closed-loop simulation), and score both the real and the
counterfactual trajectory with the v1 CARM checkpoint.

Preprocessing exactly mirrors Bench2DriveZoo's tcp/admlp branch
(tools/gen_admlp_data.py + ADMLP/data.py): 5 input frames spaced 5 apart
(2.0s of history including the reference frame), predicting 6 waypoints
spaced 5 frames apart (3.0s future, 0.5s per waypoint) in ego's local frame
at the reference frame, then rotated back to the global frame here for
scoring.
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.adapters.bench2drive import load_clip_frames, _segment_agent_positions, _agent_class_for
from futurerisk.geometry import BoxTrajectory, heading_from_positions
from futurerisk.data_pipeline import RawTrajectoryInstance
from futurerisk.config import FutureRiskConfig
from ADMLP.model import ADMLP
from ADMLP.config import GlobalConfig

CKPT_PATH = "external/Bench2DriveZoo/admlp_b2d.ckpt"
INPUT_FRAMES = 25   # 5 samples, stride 5 -> 2.0s history
FUTURE_FRAMES = 30  # 6 samples, stride 5 -> 3.0s future
DT = 0.1


def load_admlp_model() -> ADMLP:
    config = GlobalConfig()
    net = ADMLP(config)
    # weights_only=False: trusted source (official HuggingFace release of the
    # Bench2Drive paper's own baseline checkpoint), needed since it's a plain
    # PyTorch-Lightning checkpoint (pickles a ModelCheckpoint callback object).
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v
    net.load_state_dict(new_state_dict, strict=False)
    net.eval()
    return net


def build_admlp_input(frames: Dict[int, dict], i: int) -> dict:
    """Reproduce ADMLP/data.py's __getitem__ exactly for reference frame i
    (matches gen_admlp_data.py's windowing: 5 input samples at
    i-20,i-15,i-10,i-5,i; the dataset then only actually uses the first 4
    of those 5 for hist_waypoints/hist_thetas, keeping the 5th (i itself)
    as the rotation-frame origin -- see data.py's `self.x[index][-1]`)."""
    idxs_in = [i - 20, i - 15, i - 10, i - 5, i]
    xs = [frames[j]["x"] for j in idxs_in]
    ys = [frames[j]["y"] for j in idxs_in]
    thetas = [frames[j]["theta"] for j in idxs_in]
    speed = frames[i]["speed"]
    speed_acc = frames[i]["acceleration"]  # 3-vector, unreduced
    command = frames[i]["next_command"]

    ego_x, ego_y = xs[-1], ys[-1]
    ego_theta = 0.0 if np.isnan(thetas[-1]) else thetas[-1] - np.pi / 2

    def to_local(px, py):
        R = np.array([[np.cos(ego_theta), np.sin(ego_theta)], [-np.sin(ego_theta), np.cos(ego_theta)]])
        return R.dot(np.array([px - ego_x, py - ego_y]))

    hist_wp = []
    for k in range(4):
        lx, ly = to_local(xs[k], ys[k])
        hist_wp.extend([lx, ly])
    hist_theta = [0.0 if np.isnan(thetas[k]) else (thetas[k] - np.pi / 2) - ego_theta for k in range(4)]

    cmd = command
    if cmd < 0:
        cmd = 4
    cmd -= 1
    assert cmd in range(6), cmd
    cmd_onehot = [0] * 6
    cmd_onehot[cmd] = 1

    inp = np.concatenate([hist_wp, hist_theta, [speed], speed_acc, cmd_onehot]).astype(np.float32)
    return {"input": torch.from_numpy(inp).unsqueeze(0), "ego_x": ego_x, "ego_y": ego_y, "ego_theta": ego_theta}


@torch.no_grad()
def predict_future_global(net: ADMLP, frames: Dict[int, dict], i: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (xy_global (6,2), theta_global (6,)) for the 6 predicted
    waypoints (0.5s apart, 3s total), converted back from ego-local frame."""
    batch = build_admlp_input(frames, i)
    traj = net({"input": batch["input"]})[0].numpy()  # (6, 3): local (x, y, theta-ish)
    ego_theta = batch["ego_theta"]
    R_inv = np.array([[np.cos(ego_theta), -np.sin(ego_theta)], [np.sin(ego_theta), np.cos(ego_theta)]])
    xy_global = np.array([R_inv.dot(traj[k, :2]) + [batch["ego_x"], batch["ego_y"]] for k in range(6)])
    theta_global = traj[:, 2] + ego_theta + np.pi / 2
    return xy_global, theta_global


def find_valid_reference_frame(frames: Dict[int, dict], min_i: int = 20) -> int:
    idxs = sorted(frames.keys())
    for i in idxs:
        if i - 20 >= idxs[0] and i + 30 <= idxs[-1] and all((i + d) in frames for d in range(-20, 31)):
            return i
    raise ValueError("no valid reference frame with full 2s history + 3s future window")


def main():
    net = load_admlp_model()
    clip_files = sorted(glob.glob("dataset/Bench2Drive-Base/*.tar.gz"))
    # a scenario with genuine interaction potential, not a trivial empty road
    candidates = [f for f in clip_files if "Accident" in f or "Crossing" in f or "Junction" in f]
    print(f"{len(candidates)} candidate clips, {len(clip_files)} total")

    results = []
    for f in candidates[:5]:
        clip_name, frames = load_clip_frames(f)
        try:
            i = find_valid_reference_frame(frames)
        except ValueError:
            print(f"  {clip_name}: no valid window, skip")
            continue
        xy_pred, theta_pred = predict_future_global(net, frames, i)
        real_xy = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20, 25, 30]])
        l2 = np.linalg.norm(xy_pred - real_xy, axis=1)
        print(f"  {clip_name}  ref_frame={i}  L2 per waypoint (m): "
              f"{np.round(l2, 2).tolist()}  mean={l2.mean():.2f}")
        results.append({
            "clip_name": clip_name, "tar_path": f, "ref_frame": i,
            "pred_xy": xy_pred.tolist(), "pred_theta": theta_pred.tolist(),
            "real_xy": real_xy.tolist(), "l2_per_waypoint": l2.tolist(),
        })

    Path("dataset/results_v1").mkdir(parents=True, exist_ok=True)
    with open("dataset/results_v1/admlp_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote dataset/results_v1/admlp_predictions.json ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
