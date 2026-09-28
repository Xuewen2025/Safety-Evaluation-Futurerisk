#!/usr/bin/env python3
"""Run the released VAD-Base checkpoint (open-loop, no CARLA) on real
Bench2Drive scenarios and extract its predicted ego planning trajectory.
Sibling of uniad_case_study.py -- see that file's and
b2d_replay_common.py's module docstrings for the full environment and
input-construction story (both apply verbatim here; VAD and UniAD share
the same 6-camera rig, `can_bus` construction, and open-loop/no-CARLA
caveats). This file only documents what differs for VAD.

ENVIRONMENT: same as uniad_case_study.py -- run under
`external/Bench2DriveZoo/.venv_uniad`, not the main system Python:

    external/Bench2DriveZoo/.venv_uniad/bin/python \\
        scripts/experiments/vad_case_study.py

Differences from UniAD's input construction (see
`team_code/vad_b2d_agent.py`'s `VadAgent.run_step()`):
  - VAD's config (`adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py`) sets
    `model.pretrained = dict(img='ckpts/resnet50-19c8e357.pth')`, read at
    `build_model()` time to initialize the image backbone *before* the
    released full checkpoint is loaded with `strict=True` (which then
    overwrites those weights anyway) -- so `resnet50-19c8e357.pth` (also
    published on the same Hugging Face model repo) is downloaded into
    `ckpts/` purely so `build_model()` doesn't raise FileNotFoundError; it
    has no effect on the final loaded weights.
  - VAD is conditioned on the navigation command via a one-hot
    `ego_fut_cmd` (6-dim) input, not just a scalar `command` index.
  - VAD's head (`VADHead`) outputs `ego_fut_preds`: one *relative*
    (per-step displacement) trajectory per of 6 command-conditioned modes;
    `run_step()` takes `np.cumsum(..., axis=1)` to get absolute ego-local
    waypoints, then selects the mode matching the current `command`
    (exactly reproduced below).
  - `ego_lcf_feat` (9-dim "ego last-frame context" side-channel: position,
    acceleration, yaw, hardcoded ego length/width, distance-from-origin,
    previous steering command) is additionally required. Since this is a
    single isolated frame with no prior control history, the "previous
    steering command" slot is fixed at 0.0 -- exactly what the live agent
    itself has on the first control step of every real episode
    (`self.prev_control_cache` starts empty).
  - Per VAD_base_e2e_b2d.py's `inference_only_pipeline`, images are
    additionally scaled by a fixed 0.8 factor after normalization
    (`RandomScaleImageMultiViewImage(scales=[0.8])`, deterministic despite
    the name: only one scale is ever in the list) -- handled automatically
    by calling the pipeline as configured, not reimplemented here.

VAD predicts 6 waypoints; per Bench2DriveZoo's TRAIN_EVAL.md note ("VAD
calculates the average over each time period"), VAD's own convention for
what a "waypoint" represents differs subtly from UniAD's, but both are
spaced 0.5s apart over a 3.0s horizon at inference, so the same
extraction/scoring shape applies.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_replay_common import (
    EGO_LENGTH, EGO_WIDTH, build_common_results, candidate_reference_frames, load_clip_frames, local_to_global,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
B2D_ZOO_ROOT = REPO_ROOT / "external" / "Bench2DriveZoo"
CLIPS_DIR = REPO_ROOT / "dataset" / "Bench2Drive-Base"
RESULTS_DIR = REPO_ROOT / "dataset" / "results_v1"

CONFIG_PATH = B2D_ZOO_ROOT / "adzoo" / "vad" / "configs" / "VAD" / "VAD_base_e2e_b2d.py"
CKPT_PATH = B2D_ZOO_ROOT / "ckpts" / "vad_b2d_base.pth"

# See uniad_case_study.py's identical comment: the config's own relative
# paths ('ckpts/resnet50-19c8e357.pth', etc.) are relative to
# Bench2DriveZoo's repo root, so just run from there.
os.chdir(B2D_ZOO_ROOT)


def load_vad_model():
    from mmcv import Config
    from mmcv.datasets.pipelines import Compose
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint

    cfg = Config.fromfile(str(CONFIG_PATH))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(CKPT_PATH), map_location='cpu', strict=True)
    model.cuda()
    model.eval()

    pipeline = [p for p in cfg.inference_only_pipeline if p['type'] not in ('LoadMultiViewImageFromFilesInCeph',
                                                                             'LoadMultiViewImageFromFiles')]
    return model, Compose(pipeline)


def _add_vad_fields(results: dict) -> dict:
    """Add VAD-only fields (`ego_fut_cmd`, `ego_lcf_feat`) to the
    model-agnostic `results` dict, mirroring VadAgent.run_step()."""
    can_bus = results['can_bus']
    command = results['command']
    command_onehot = np.zeros(6)
    command_onehot[command] = 1
    results['ego_fut_cmd'] = command_onehot

    rotation = list(can_bus[3:7])
    ego_lcf_feat = np.zeros(9)
    ego_lcf_feat[0:2] = can_bus[0:2].copy()
    ego_lcf_feat[2:4] = can_bus[10:12].copy()
    ego_lcf_feat[4] = rotation[-1]
    ego_lcf_feat[5] = EGO_LENGTH
    ego_lcf_feat[6] = EGO_WIDTH
    ego_lcf_feat[7] = np.sqrt(can_bus[0] ** 2 + can_bus[1] ** 2)
    ego_lcf_feat[8] = 0.0  # no prior control step for an isolated single frame (see module docstring)
    results['ego_lcf_feat'] = ego_lcf_feat
    return results


@torch.no_grad()
def predict_future_global(model, pipeline, tar_path: str, frames: dict, i: int):
    """Returns xy_global (6, 2): VAD's 6 predicted, command-conditioned
    planning waypoints (0.5s apart, 3.0s total), converted from ego-local
    back to the global frame."""
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form

    results = build_common_results(tar_path, frames, i)
    ego_theta, ego_x, ego_y = results.pop('_ego_theta'), results.pop('_ego_x'), results.pop('_ego_y')
    command = results['command']
    results = _add_vad_fields(results)
    results = pipeline(results)
    batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
    for key, data in batch.items():
        if key != 'img_metas' and torch.is_tensor(data[0]):
            data[0] = data[0].to('cuda')

    out = model(batch, return_loss=False, rescale=True)
    all_out_truck_d1 = out[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()  # (6 modes, 6 steps, 2)
    all_out_truck = np.cumsum(all_out_truck_d1, axis=1)
    local_wp_raw = all_out_truck[command]  # (6, 2), mode matching the current nav command
    # Same empirically-verified column swap as uniad_case_study.py's
    # `sdc_traj` -- `ego_fut_preds`' two columns are likewise ordered
    # (lateral, forward), not (forward, lateral); see that file's comment
    # for the verification method (this was checked independently for VAD,
    # not assumed to carry over from UniAD).
    local_wp = local_wp_raw[:, ::-1]
    xy_global = local_to_global(local_wp, ego_x, ego_y, ego_theta)
    return xy_global, local_wp


def main():
    model, pipeline = load_vad_model()
    clip_files = sorted(glob.glob(str(CLIPS_DIR / "*.tar.gz")))
    candidates = [f for f in clip_files if any(k in f for k in ("Accident", "Crossing", "Junction", "Cut"))]
    print(f"{len(candidates)} candidate clips, {len(clip_files)} total")

    # Same rationale as uniad_case_study.py: scan several candidate
    # reference frames per clip and keep the one with the largest L2, since
    # nominal driving is predicted accurately (~1-3m scale, see
    # dataset/results_v1/uniad_vad_setup_notes.md) and an interesting
    # divergence for the counterfactual case study is more likely deeper
    # into a scenario's hazard.
    results_out = []
    for f in candidates[:15]:
        clip_name, frames = load_clip_frames(f)
        ref_frames = candidate_reference_frames(frames)
        if not ref_frames:
            print(f"  {clip_name}: no valid window, skip")
            continue
        best = None
        for i in ref_frames:
            xy_pred, _ = predict_future_global(model, pipeline, f, frames, i)
            real_xy = np.array([[frames[i + d]["x"], frames[i + d]["y"]] for d in [5, 10, 15, 20, 25, 30]])
            l2 = np.linalg.norm(xy_pred - real_xy, axis=1)
            if best is None or l2.mean() > best[2]:
                best = (i, xy_pred, l2.mean(), real_xy, l2)
        i, xy_pred, _, real_xy, l2 = best
        print(f"  {clip_name}  ref_frame={i}  L2 per waypoint (m): "
              f"{np.round(l2, 2).tolist()}  mean={l2.mean():.2f}")
        results_out.append({
            "clip_name": clip_name, "tar_path": f, "ref_frame": i,
            "pred_xy": xy_pred.tolist(), "real_xy": real_xy.tolist(), "l2_per_waypoint": l2.tolist(),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "vad_predictions.json"
    with open(out_path, "w") as fh:
        json.dump(results_out, fh, indent=2)
    print(f"wrote {out_path} ({len(results_out)} scenarios)")


if __name__ == "__main__":
    main()
