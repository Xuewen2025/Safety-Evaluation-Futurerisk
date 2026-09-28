#!/usr/bin/env python3
"""Run the released UniAD-Tiny checkpoint (open-loop, no CARLA) on real
Bench2Drive scenarios and extract its predicted ego planning trajectory --
the uniad/vad-branch analogue of tcp_case_study.py's
`predict_future_global`, for a full surround-view camera + BEV planner
instead of TCP's single front-camera model.

ENVIRONMENT: unlike tcp_case_study.py/admlp_case_study.py, this script does
NOT run in the main system Python. UniAD/VAD (Bench2DriveZoo's `uniad/vad`
git branch) merge mmcv+mmseg+mmdet+mmdet3d into one package with compiled
CUDA extensions, and pin dependency versions (numpy==1.20.0, numba==0.48.0)
that predate Python 3.10 wheels. It runs inside a dedicated virtualenv,
`external/Bench2DriveZoo/.venv_uniad` (see
dataset/results_v1/uniad_vad_setup_notes.md for exactly how that
environment was built and every version deviation from the repo's stated
install instructions), which has no `futurerisk`. Invoke it as:

    external/Bench2DriveZoo/.venv_uniad/bin/python \\
        scripts/experiments/uniad_case_study.py

This is stage A of a two-stage pipeline (mirrors tcp_/admlp_ case studies'
single-environment design, but split across environments): stage A (this
script) writes predicted trajectories to
dataset/results_v1/uniad_predictions.json; stage B
(uniad_counterfactual_case_study.py, run under the MAIN system Python) reads
that JSON and does the CARM v1 counterfactual scoring, exactly like
tcp_counterfactual_case_study.py does for TCP. The two stages cannot share a
process since they need incompatible Python environments.

Model input construction: see scripts/experiments/b2d_replay_common.py's
module docstring for the full, detailed account of what is replayed
verbatim from `team_code/uniad_b2d_agent.py`'s `UniadAgent.setup()`/
`run_step()` vs. reconstructed from one real clip's recorded `anno`
JSON + saved camera jpgs (no CARLA, no live sensors) -- including the one
place (world-frame ego heading) where this reconstruction is a documented,
unverified best-effort assumption rather than a directly-recovered
quantity.

UniAD's released checkpoint (`uniad_tiny_b2d.pth`) is loaded with
`strict=True` (exactly as `UniadAgent.setup()` does), so no separate
backbone-pretrain weight (`resnet50-19c8e357.pth`) is needed for inference
-- the released checkpoint already contains the fully fine-tuned backbone.
`data/others/b2d_motion_anchor_infos_mode6.pkl` (needed to *construct* the
model architecture: `MotionHead.__init__` loads it directly) was already
present in this repo checkout, not downloaded separately.

UniAD's planning head (`PlanningHeadSingleMode`, `planning_steps=6`)
predicts a single 6-waypoint ego trajectory (no navigation-command
conditioning branch, unlike VAD) in the ego/LiDAR frame at the reference
frame; `output['planning']['result_planning']['sdc_traj']` is that
trajectory, spaced 0.5s apart (3.0s total) matching the training config's
`planning_steps=6` at Bench2Drive's 0.5s-per-waypoint convention (same as
AD-MLP's horizon, longer than TCP's 2.0s).
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
    build_common_results, candidate_reference_frames, load_clip_frames, local_to_global,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
B2D_ZOO_ROOT = REPO_ROOT / "external" / "Bench2DriveZoo"
CLIPS_DIR = REPO_ROOT / "dataset" / "Bench2Drive-Base"
RESULTS_DIR = REPO_ROOT / "dataset" / "results_v1"

CONFIG_PATH = B2D_ZOO_ROOT / "adzoo" / "uniad" / "configs" / "stage2_e2e" / "tiny_e2e_b2d.py"
CKPT_PATH = B2D_ZOO_ROOT / "ckpts" / "uniad_tiny_b2d.pth"

# The config's `motion_head.anchor_info_path` ('data/others/...') is a path
# relative to Bench2DriveZoo's own repo root (that's where the live
# UniadAgent expects to be run from, one level up in the CARLA leaderboard
# layout, with an extra 'Bench2DriveZoo/' prefix it adds itself -- see
# team_code/uniad_b2d_agent.py). Simpler here: just run from that root
# directly, so the path in the config resolves unmodified.
os.chdir(B2D_ZOO_ROOT)


def load_uniad_model():
    # Imports deferred to inside this function: only importable once run
    # under external/Bench2DriveZoo/.venv_uniad (see module docstring).
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


@torch.no_grad()
def predict_future_global(model, pipeline, tar_path: str, frames: dict, i: int):
    """Returns xy_global (6, 2): UniAD's 6 predicted planning waypoints
    (0.5s apart, 3.0s total), converted from ego-local back to the global
    frame. Mirrors tcp_case_study.py's `predict_future_global` in shape and
    intent."""
    from mmcv.core.bbox import get_box_type  # noqa: F401  (imported for parity/side effects)
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form

    results = build_common_results(tar_path, frames, i)
    ego_theta, ego_x, ego_y = results.pop('_ego_theta'), results.pop('_ego_x'), results.pop('_ego_y')
    results = pipeline(results)
    batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
    for key, data in batch.items():
        if key != 'img_metas' and torch.is_tensor(data[0]):
            data[0] = data[0].to('cuda')

    out = model(batch, return_loss=False, rescale=True)
    local_wp_raw = out[0]['planning']['result_planning']['sdc_traj'][0].cpu().numpy()  # (6, 2)
    # `sdc_traj`'s two columns are NOT (x=forward, y=left) as the symmetric
    # `point_cloud_range`/nuScenes-LiDAR convention would suggest -- verified
    # empirically (see b2d_replay_common.py's module docstring for the same
    # kind of check applied to `ego_theta`): rotating a real clip's own
    # future displacement into the `ego_theta` frame puts the dominant,
    # monotonically-growing "drove forward" component in index 0 and a
    # small residual in index 1, whereas `sdc_traj` has that pattern
    # reversed (index 1 dominant/growing, index 0 near zero). Swapping
    # columns before rotating back to global fixes this; this is the one
    # place this script deviates from a literal port of
    # `UniadAgent.run_step()` (which never needs to interpret `sdc_traj` in
    # the global frame -- it hands it directly to a PID controller in the
    # ego frame, so this axis convention never had to be pinned down there).
    local_wp = local_wp_raw[:, ::-1]
    xy_global = local_to_global(local_wp, ego_x, ego_y, ego_theta)
    return xy_global, local_wp


def main():
    model, pipeline = load_uniad_model()
    clip_files = sorted(glob.glob(str(CLIPS_DIR / "*.tar.gz")))
    candidates = [f for f in clip_files if any(k in f for k in ("Accident", "Crossing", "Junction", "Cut"))]
    print(f"{len(candidates)} candidate clips, {len(clip_files)} total")

    # For each clip, scan several candidate reference frames (every 20th
    # frame with a full future window and non-degenerate ego speed) and
    # keep the one with the largest predicted-vs-real L2 -- same rationale
    # as tcp_counterfactual_case_study.py: nominal driving is predicted
    # accurately enough (see dataset/results_v1/uniad_vad_setup_notes.md
    # for the ~1m-scale L2 this model achieves on ordinary frames) that an
    # interesting divergence worth a counterfactual case study is more
    # likely deeper into a scenario's hazard than at an arbitrary frame.
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
    out_path = RESULTS_DIR / "uniad_predictions.json"
    with open(out_path, "w") as fh:
        json.dump(results_out, fh, indent=2)
    print(f"wrote {out_path} ({len(results_out)} scenarios)")


if __name__ == "__main__":
    main()
