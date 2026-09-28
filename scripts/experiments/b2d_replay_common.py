#!/usr/bin/env python3
"""Shared "replay one real Bench2Drive clip through a released camera-based
Bench2DriveZoo planner, offline" helpers for uniad_case_study.py and
vad_case_study.py.

IMPORTANT: this module runs inside the `external/Bench2DriveZoo/.venv_uniad`
virtualenv (it imports `mmcv`/`pyquaternion` from the uniad/vad branch's own
install, not from the main system Python that hosts `futurerisk`). It has no
dependency on `futurerisk` and must not be imported from the main
environment. See uniad_case_study.py's module docstring for the full
environment/isolation story.

What this reproduces, and from where
-------------------------------------
`team_code/uniad_b2d_agent.py` (`UniadAgent`) and `team_code/vad_b2d_agent.py`
(`VadAgent`) are CARLA-leaderboard agents: `setup()` builds the model and
loads the released checkpoint, `sensors()` declares a fixed 6-camera +
IMU + GPS + speedometer rig, and `run_step(input_data, timestamp)` is called
once per simulation tick with live sensor readings, builds a `results` dict,
runs it through `cfg.inference_only_pipeline`, collates it into a batch, and
calls the model. Both agents' `run_step` are near-identical (same 6-camera
extrinsics, same `can_bus` construction); this module factors out that
shared, model-agnostic part so it does not need to be duplicated per model.

Since there is no live CARLA/leaderboard session here, every quantity that
`run_step` would read off a live sensor is instead read from one already
-recorded Bench2Drive clip's `anno/<frame>.json.gz` (the same per-frame
ground-truth JSON that scripts/experiments/tcp_case_study.py and
admlp_case_study.py already use) and its saved `camera/rgb_*/<frame>.jpg`
files -- i.e. this replays one real frame's already-recorded sensor state
through the model exactly once, open-loop, no CARLA. Concretely:

- The 6 camera images: loaded directly from the clip's
  `camera/rgb_{front,front_left,front_right,back,back_left,back_right}/
  <frame>.jpg` (same convention already verified working by
  tcp_case_study.py's `load_frame_images`, extended here to all 6 views).
  Confirmed 1600x900 RGB, matching the `sensors()` declaration's
  `width=1600, height=900` for every CAM_* sensor.
- `lidar2img` / `lidar2cam` / `lidar2ego`: copied verbatim from
  `UniadAgent.setup()` / `VadAgent.setup()` (byte-for-byte identical between
  the two). These are NOT per-frame quantities -- Bench2Drive uses one fixed
  camera rig for the whole dataset, so hardcoding them (exactly as the
  released agents themselves do) is the faithful choice, not an
  approximation.
- `speed`, `acceleration` (3-vector), `angular_velocity` (3-vector): read
  directly from the anno JSON's own `speed`/`acceleration`/`angular_velocity`
  fields, which is exactly what a live IMU/speedometer would have supplied
  to `tick()` -- these are the *same underlying simulator quantities*, not a
  re-derivation.
- `command` (next high-level navigation command): `anno['next_command']`,
  remapped exactly like tcp_case_study.py/admlp_case_study.py already do
  (`VOID`/negative -> LANEFOLLOW, then -1 to 0-index into 6 classes).
- World-frame ego position/heading fed into `can_bus` (position, rotation
  quaternion) and into `ego2world`/`l2g_r_mat`/`l2g_t`: there is no live GPS
  or compass reading to replay, so this uses the clip's own `anno['x']`,
  `anno['y']`, and `ego_theta = anno['theta'] - pi/2` -- the SAME rotation
  TCP/AD-MLP already use (see tcp_case_study.py/admlp_case_study.py) to
  convert world offsets into their own ego-local frame, NOT the live
  UniAD/VAD agent's own `ego_theta = -compass + pi/2` compass-inversion
  formula (there is no live compass reading to invert here).
    * This was verified empirically, not assumed: for a real reference
      frame, rotating the clip's own real future ego displacement into
      "ego_theta" frame via `theta` directly produces a local trajectory
      whose *dominant* growth is in the lateral (y) component -- physically
      implausible for a car driving along a road -- while `theta - pi/2`
      produces the physically-sensible pattern (dominant growth along
      local +x, i.e. "forward", with a small lateral residual). Whatever
      `anno['theta']`'s exact underlying convention is, `theta - pi/2` is
      empirically the rotation that aligns it with the ego/LiDAR frame's
      forward axis, consistent with (and reusing) this codebase's own
      already-established convention rather than a fresh derivation.
    * `futurerisk`'s own Bench2Drive adapter
      (`futurerisk/adapters/bench2drive.py`) separately documents being
      unable to fully pin down this field's convention either, and derives
      heading from box-center motion instead for its own (collision
      -geometry) purposes -- a different use case that doesn't need an
      ego heading *before* any predicted future exists, unlike this module.
    * This choice feeds the model's auxiliary `can_bus`/`ego_lcf_feat`
      ego-motion side-channel and the (largely moot, for a single isolated
      first frame with no prior BEV state) `l2g_r_mat`/`l2g_t` temporal
      -alignment inputs -- NOT the core camera-based perception/planning
      geometry itself, which is computed entirely in the ego/LiDAR frame
      via the fixed extrinsics above and is unaffected by this choice.
    * The SAME `ego_theta` is reused to rotate the model's predicted
      ego-local-frame trajectory back into the global frame for scoring,
      exactly as tcp_case_study.py/admlp_case_study.py already do. Both
      uniad_case_study.py and vad_case_study.py print predicted-vs-real L2
      error for exactly this reason, so a wrong assumption here is visible
      in the printed output (a large, systematically-growing error) rather
      than silently baked into results_v1/*.json.
- `timestamp`, `frame_idx`, `scene_token`, `folder`: dummy placeholders
  (`0.0`, `0`, `' '`, `' '`), matching what `run_step` sets for the *first*
  tick of a fresh episode (`self.step` starts at -1, so the first call has
  `timestamp = 0/20 = 0.0`) -- this replay is exactly that "first tick of a
  fresh episode" code path (a fresh `build_model()` per call, no prior
  `prev_bev`), which both released agents also go through at the start of
  every real closed-loop episode.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

CAM_KEYS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAM_FOLDER = {
    "CAM_FRONT": "rgb_front", "CAM_FRONT_LEFT": "rgb_front_left", "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back", "CAM_BACK_LEFT": "rgb_back_left", "CAM_BACK_RIGHT": "rgb_back_right",
}

# ---------------------------------------------------------------------------
# Fixed sensor rig, copied verbatim from team_code/uniad_b2d_agent.py's (==
# vad_b2d_agent.py's) UniadAgent.setup() / VadAgent.setup(). Dataset-constant
# across every Bench2Drive clip -- not derived or fitted here.
# ---------------------------------------------------------------------------
LIDAR2IMG = {
    'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK': np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
}
LIDAR2CAM = {
    'CAM_FRONT': np.array([[1., 0., 0., 0.],
                           [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19],
                           [0., 0., 0., 1.]]),
    'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),
    'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),
    'CAM_BACK': np.array([[-1., 0., 0., 0.],
                          [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61],
                          [0., 0., 0., 1.]]),
    'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),
    'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]]),
}
LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

# Ego vehicle half-extents used by VAD's ego_lcf_feat (hardcoded identically
# in vad_b2d_agent.py; not read from anno since ego's own box extent is
# always this same fixed vehicle model across the dataset).
EGO_LENGTH = 4.89238167
EGO_WIDTH = 1.83671331


def load_clip_frames(tar_path: str) -> Tuple[str, Dict[int, dict]]:
    """Standalone re-implementation of
    futurerisk.adapters.bench2drive.load_clip_frames (anno-only, one pass).
    Duplicated (not imported) because this module runs in the isolated
    uniad/vad venv, which does not have `futurerisk` installed -- see this
    file's module docstring."""
    frames: Dict[int, dict] = {}
    clip_name = None
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or "/anno/" not in member.name or not member.name.endswith(".json.gz"):
                continue
            if clip_name is None:
                clip_name = member.name.split("/anno/")[0].lstrip("./")
            fname = member.name.rsplit("/", 1)[-1]
            try:
                idx = int(fname.split(".")[0])
            except ValueError:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            with gzip.GzipFile(fileobj=fobj) as gz:
                frames[idx] = json.loads(gz.read())
    return clip_name, frames


def load_frame_images_6cam(tar_path: str, idx: int) -> Dict[str, np.ndarray]:
    """Extract all 6 surround-view jpgs for one reference frame, RGB uint8
    arrays straight from PIL (no color-space conversion needed: the live
    agent's tick() does cv2.cvtColor(BGR2RGB) on its raw camera buffer
    before use, and these already-saved jpgs are RGB on disk -- verified
    directly, same as tcp_case_study.py's load_frame_images)."""
    wanted = {f"camera/{folder}/{idx:05d}.jpg": cam for cam, folder in CAM_FOLDER.items()}
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            for suffix, cam in wanted.items():
                if member.name.endswith(suffix):
                    fobj = tar.extractfile(member)
                    out[cam] = np.array(Image.open(io.BytesIO(fobj.read())))
                    break
    missing = set(wanted.values()) - set(out.keys())
    if missing:
        raise ValueError(f"missing camera frames {missing} for idx={idx} in {tar_path}")
    return out


def build_common_results(tar_path: str, frames: Dict[int, dict], i: int) -> dict:
    """Build the model-agnostic parts of the `results` dict that
    UniadAgent.run_step()/VadAgent.run_step() populate before calling
    `self.inference_only_pipeline(results)`. Model-specific keys
    (`ego_fut_cmd`, `ego_lcf_feat` for VAD) are added by the caller."""
    from mmcv.core.bbox import get_box_type  # deferred: only importable inside the venv

    anno = frames[i]
    imgs = load_frame_images_6cam(tar_path, i)

    results: dict = {
        'lidar2img': np.stack([LIDAR2IMG[c] for c in CAM_KEYS], axis=0),
        'lidar2cam': np.stack([LIDAR2CAM[c] for c in CAM_KEYS], axis=0),
        'img': [imgs[c] for c in CAM_KEYS],
        'folder': ' ',
        'scene_token': ' ',
        'frame_idx': 0,
        'timestamp': 0.0,
    }
    results['box_type_3d'], _ = get_box_type('LiDAR')

    # See module docstring: ego_theta = anno['theta'] - pi/2, matching
    # TCP/AD-MLP's own established local-frame convention (empirically
    # verified here, not merely assumed).
    ego_theta = -np.pi / 2 if np.isnan(anno['theta']) else float(anno['theta']) - np.pi / 2
    rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

    can_bus = np.zeros(18)
    can_bus[0] = anno['x']
    can_bus[1] = -anno['y']
    can_bus[3:7] = rotation
    can_bus[7] = anno['speed']
    can_bus[10:13] = anno['acceleration']
    can_bus[11] *= -1
    can_bus[13:16] = -np.array(anno['angular_velocity'])
    can_bus[16] = ego_theta
    can_bus[17] = ego_theta / np.pi * 180
    results['can_bus'] = can_bus

    command = anno['next_command']
    if command < 0:
        command = 4
    command -= 1
    assert command in range(6), command
    results['command'] = command

    ego2world = np.eye(4)
    ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
    ego2world[0:2, 3] = can_bus[0:2]
    lidar2global = ego2world @ LIDAR2EGO
    results['l2g_r_mat'] = lidar2global[0:3, 0:3]
    results['l2g_t'] = lidar2global[0:3, 3]

    stacked_imgs = np.stack(results['img'], axis=-1)
    results['img_shape'] = stacked_imgs.shape
    results['ori_shape'] = stacked_imgs.shape
    results['pad_shape'] = stacked_imgs.shape

    results['_ego_theta'] = ego_theta
    results['_ego_x'] = float(anno['x'])
    results['_ego_y'] = float(anno['y'])
    return results


def local_to_global(local_xy: np.ndarray, ego_x: float, ego_y: float, ego_theta: float) -> np.ndarray:
    """Rotate an ego-local-frame (forward=+x, left=+y -- standard nuScenes/
    mmdet3d LiDAR convention, matching this repo's `box_type_3d='LiDAR'` and
    symmetric point_cloud_range) trajectory back into the global frame,
    using the SAME ego_theta convention build_common_results() assumed.
    Identical rotation-matrix form to tcp_case_study.py/admlp_case_study.py
    (`R_inv`), just applied to whatever number of waypoints `local_xy` has."""
    c, s = np.cos(ego_theta), np.sin(ego_theta)
    r_inv = np.array([[c, -s], [s, c]])
    return np.array([r_inv.dot(xy) + [ego_x, ego_y] for xy in local_xy])


def find_valid_reference_frame(
    frames: Dict[int, dict], future_frames: int = 30, min_i: int = 20, min_speed: float = 1.0,
) -> int:
    """First frame index with a full `future_frames`-long contiguous window
    after it (default 30 frames = 3.0s at 10Hz, matching both UniAD's
    planning_steps=6 and VAD's valid_fut_ts=6 at their trained 0.5s
    waypoint spacing) -- needed only to compare the prediction against real
    future ego positions, not by the model itself (a single frame's images
    are all either model needs).

    Skips frame 0 (and the first `min_i` frames): confirmed empirically
    that clip-start frames have ego essentially stationary (speed ~1e-5
    m/s in every clip checked) -- a fresh-episode spawn point, not a
    representative driving state, and a materially harder prediction
    target (a stopped car gives the model no motion cues to extrapolate
    from). `min_speed` enforces this directly rather than just skipping a
    fixed frame count, since some clips stay stopped (e.g. at a red light)
    well past frame `min_i`."""
    idxs = sorted(frames.keys())
    for i in idxs:
        if i < min_i:
            continue
        if frames[i].get("speed", 0.0) < min_speed:
            continue
        if i + future_frames <= idxs[-1] and all((i + d) in frames for d in range(0, future_frames + 1)):
            return i
    raise ValueError(f"no valid reference frame with a full {future_frames}-frame future window and speed>={min_speed}")


def candidate_reference_frames(
    frames: Dict[int, dict], future_frames: int = 30, min_i: int = 20, min_speed: float = 1.0, stride: int = 20,
) -> List[int]:
    """Every `stride`-th frame with a full future window and non-degenerate
    (moving) ego speed -- same idea as
    tcp_counterfactual_case_study.py/admlp_counterfactual_case_study.py's
    own `candidate_reference_frames`, generalized with the same
    stationary-frame filter as find_valid_reference_frame() above."""
    idxs = sorted(frames.keys())
    out = []
    for i in idxs[min_i::stride]:
        if frames[i].get("speed", 0.0) < min_speed:
            continue
        if i + future_frames <= idxs[-1] and all((i + d) in frames for d in range(0, future_frames + 1)):
            out.append(i)
    return out
