#!/usr/bin/env python3
"""Run the released TCP checkpoint (open-loop, no CARLA) on real Bench2Drive
scenarios: same "what would FutureRisk have scored if this policy had been
driving" comparison as admlp_case_study.py, but for TCP, which (unlike
AD-MLP) conditions on the actual front-camera image rather than only ego
kinematics.

Preprocessing exactly mirrors Bench2DriveZoo's tcp/admlp branch (TCP/data.py
CARLA_Data.__getitem__): concatenate front_left[:, :1400] + front[:, 200:1400]
+ front_right[:, 200:] horizontally, resize to (256, 900), ImageNet-normalize;
ego_theta = anno['theta'] - pi/2; target_point = the far navigation command
point (x_command_far/y_command_far... actually anno's own x_target/y_target,
see below) rotated into the ego frame; speed fed in as anno['speed']/12;
command is anno['next_command'] (VOID -> LANEFOLLOW), one-hot 6-dim. TCP
predicts 4 waypoints at 0.5s spacing (2.0s total) autoregressively via a GRU
decoder conditioned on target_point -- a shorter horizon than AD-MLP's 3s.
"""
from __future__ import annotations

import glob
import io
import json
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "Bench2DriveZoo"))

from futurerisk.adapters.bench2drive import load_clip_frames
from TCP.model import TCP
from TCP.config import GlobalConfig

CKPT_PATH = "external/Bench2DriveZoo/tcp_b2d.ckpt"
_IM_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_tcp_model() -> TCP:
    config = GlobalConfig()
    net = TCP(config)
    # weights_only=False: trusted source (official HuggingFace release of the
    # Bench2Drive paper's own TCP baseline checkpoint), a PyTorch-Lightning
    # checkpoint (pickles a ModelCheckpoint callback object).
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v
    net.load_state_dict(new_state_dict, strict=False)
    net.eval()
    return net


def load_frame_images(tar_path: str, idx: int) -> Dict[str, np.ndarray]:
    """Extract only the 3 front-facing jpgs needed for one reference frame
    (front, front_left, front_right), without touching any other tar member."""
    wanted = {
        f"camera/rgb_front/{idx:05d}.jpg": "front",
        f"camera/rgb_front_left/{idx:05d}.jpg": "front_left",
        f"camera/rgb_front_right/{idx:05d}.jpg": "front_right",
    }
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            for suffix, key in wanted.items():
                if member.name.endswith(suffix):
                    fobj = tar.extractfile(member)
                    out[key] = np.array(Image.open(io.BytesIO(fobj.read())))
                    break
    missing = set(wanted.values()) - set(out.keys())
    if missing:
        raise ValueError(f"missing camera frames {missing} for idx={idx} in {tar_path}")
    return out


def build_tcp_input(frames: Dict[int, dict], tar_path: str, i: int) -> dict:
    """Reproduce TCP/data.py's CARLA_Data.__getitem__ exactly for reference
    frame i (single input frame, INPUT_FRAMES=1)."""
    imgs = load_frame_images(tar_path, i)
    front_img = imgs["front"][:, 200:1400, :]
    front_left_img = imgs["front_left"][:, :1400, :]
    front_right_img = imgs["front_right"][:, 200:, :]
    cat = np.concatenate((front_left_img, front_img, front_right_img), axis=1)
    cat_t = torch.from_numpy(cat).permute(2, 0, 1).unsqueeze(0).float()
    cat_t = F.interpolate(cat_t, size=(256, 900), mode="bilinear", align_corners=False)
    cat_np = cat_t.squeeze(0).permute(1, 2, 0).byte().numpy()
    front_img_tensor = _IM_TRANSFORM(np.array(cat_np)).unsqueeze(0)

    anno = frames[i]
    theta = anno["theta"]
    ego_x, ego_y = anno["x"], anno["y"]
    ego_theta = 0.0 if np.isnan(theta) else theta - np.pi / 2

    R = np.array([[np.cos(ego_theta), np.sin(ego_theta)], [-np.sin(ego_theta), np.cos(ego_theta)]])
    target_point = R.dot(np.array([anno["x_target"] - ego_x, anno["y_target"] - ego_y]))

    command = anno["next_command"]
    if command < 0:
        command = 4
    command -= 1
    assert command in range(6), command
    cmd_onehot = [0.0] * 6
    cmd_onehot[command] = 1.0

    speed = torch.tensor([[anno["speed"] / 12.0]], dtype=torch.float32)
    target_point_t = torch.tensor([target_point], dtype=torch.float32)
    command_t = torch.tensor([cmd_onehot], dtype=torch.float32)
    state = torch.cat([speed, target_point_t, command_t], dim=1)

    return {
        "front_img": front_img_tensor, "state": state, "target_point": target_point_t,
        "ego_x": ego_x, "ego_y": ego_y, "ego_theta": ego_theta,
    }


@torch.no_grad()
def predict_future_global(net: TCP, frames: Dict[int, dict], tar_path: str, i: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns xy_global (4, 2) for the 4 predicted waypoints (0.5s apart,
    2.0s total), converted back from ego-local frame. TCP's decoder does not
    predict heading, so only positions are returned."""
    batch = build_tcp_input(frames, tar_path, i)
    out = net(batch["front_img"], batch["state"], batch["target_point"])
    local_wp = out["pred_wp"][0].numpy()  # (4, 2), ego-local (left-hand rotated frame)
    ego_theta = batch["ego_theta"]
    R_inv = np.array([[np.cos(ego_theta), -np.sin(ego_theta)], [np.sin(ego_theta), np.cos(ego_theta)]])
    xy_global = np.array([R_inv.dot(local_wp[k]) + [batch["ego_x"], batch["ego_y"]] for k in range(4)])
    return xy_global, local_wp


def find_valid_reference_frame(frames: Dict[int, dict], min_i: int = 5) -> int:
    idxs = sorted(frames.keys())
    for i in idxs:
        if i + 20 <= idxs[-1] and all((i + d) in frames for d in range(0, 21)):
            return i
    raise ValueError("no valid reference frame with a full 2s future window")


def main():
    net = load_tcp_model()
    clip_files = sorted(glob.glob("dataset/Bench2Drive-Base/*.tar.gz"))
    candidates = [f for f in clip_files if any(k in f for k in ("Accident", "Crossing", "Junction"))]
    print(f"{len(candidates)} candidate clips, {len(clip_files)} total")

    results = []
    for f in candidates[:5]:
        clip_name, frames = load_clip_frames(f)
        try:
            i = find_valid_reference_frame(frames)
        except ValueError:
            print(f"  {clip_name}: no valid window, skip")
            continue
        xy_pred, local_wp = predict_future_global(net, frames, f, i)
        real_xy = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20]])
        l2 = np.linalg.norm(xy_pred - real_xy, axis=1)
        print(f"  {clip_name}  ref_frame={i}  L2 per waypoint (m): "
              f"{np.round(l2, 2).tolist()}  mean={l2.mean():.2f}")
        results.append({
            "clip_name": clip_name, "tar_path": f, "ref_frame": i,
            "pred_xy": xy_pred.tolist(), "real_xy": real_xy.tolist(), "l2_per_waypoint": l2.tolist(),
        })

    Path("dataset/results_v1").mkdir(parents=True, exist_ok=True)
    with open("dataset/results_v1/tcp_predictions.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote dataset/results_v1/tcp_predictions.json ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
