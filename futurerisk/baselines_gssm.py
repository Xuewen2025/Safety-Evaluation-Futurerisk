"""GSSM (Generalised Surrogate Safety Measure, Jiao et al. 2026, Nature
Machine Intelligence; arXiv:2505.13556) -- a third comparable-effort,
non-fabricated reproduction of a third-party learning-based safety
baseline, alongside `baselines_extra.LuEtAlReproduction` (same rigor and
documentation convention as that module: read it first).

Path taken: RELEASED CHECKPOINT, ZERO-SHOT (not retrained). GSSM's authors
publish trained checkpoints and per-configuration validation losses at
https://doi.org/10.4121/9caa1e6c-9abd-4e36-ae28-c9ea4542d940 (downloaded
and verified against the archive's own `computed_md5` here). The checkpoint
used is `PosteriorInference/SafeBaseline/current+acc/model_final_71epoch.pth`
-- trained on SHRP2 NDS naturalistic-driving events ("SafeBaseline" in the
paper's own dataset naming) using ONLY the single-encoder `current+acc`
input (13 kinematic features per ego/other-agent pair per frame -- no
environment (weather/lighting/road-surface) or historical-profile
features). That specific configuration was chosen, out of the six
released for SafeBaseline alone, because it is the only one whose full
input can be reconstructed from this project's own dataset-agnostic
`BoxTrajectory` fields (position, heading, extent) with no invented or
imputed data: `current+acc_environment`/`..._environment_profiles` need
SHRP2-specific categorical environment labels (lighting/weather/surface/
traffic density) that DeepAccident and Bench2Drive do not carry in a
comparable form, and `profiles` needs a 2.5s kinematic history per frame
that would have to be resampled/invented for the first 2.5s of every
instance. Using `current+acc` zero-shot is thus the same "train once on
one domain, evaluate zero-shot elsewhere" protocol this project already
applies to CARM (DeepAccident -> Bench2Drive), except here the "training
domain" is the paper's own SHRP2 NDS release rather than DeepAccident.

Model, exact input features, and the risk score
-------------------------------------------------
GSSM predicts a context-conditioned log-normal distribution
log(spacing) ~ N(mu, sigma^2) for the Euclidean center-to-center spacing
`s` between ego and one other agent at one frame, conditioned on 13
"current+acc" features (see `_CURRENT_ACC_COLUMNS` below and
`gssm_current_acc_features` for the exact per-frame construction, ported
directly from the official repo's
`src_data_preparation/represent_utils/utils_data_segmentation.py`
(`ContextSegmenter.segment_data`) and
`src_data_preparation/represent_utils/coortrans.py` (`rotate_coor`,
`angle`) -- the same rotation/angle helpers are reproduced verbatim here
in numpy, not re-derived from the paper text, specifically to avoid a
sign-convention transcription error). The model itself is never
retrained here: no collision label, no fine-tuning, weights exactly as
released.

The risk score ("intensity") is NOT the log-normal density and NOT mu or
sigma directly. It answers: given the model's predicted "typical spacing"
distribution for this context, how extreme (close) is the spacing
actually observed at this frame, expressed as log10(n) where n is the
number of i.i.d. draws from that context's distribution that would be
needed for the minimum of n draws to have a 50% chance of being at least
this close. This is GSSM's own `torch_intensity` function (paper's "return
period"-style extremity score, used for their own conflict/warning
detection in `evaluate_safety.py`/`risk_evaluation.py`), imported
UNMODIFIED from the cloned official repo inside the isolated venv (see
`external/GSSM/gssm_infer.py`) -- reproduced here in one place, not
re-derived independently, to avoid any transcription error in the one
formula this baseline's numbers hinge on. Higher intensity = the observed
spacing is a deeper excursion into the close tail of "normal" driving for
this context = riskier.

Environment isolation
----------------------
GSSM's own code (`external/GSSM/`, MIT-licensed upstream clone) requires
torch==2.6.0 and Python 3.10 pinned by its own `pyproject.toml`, which the
system Python's torch 2.14+cu130 need not match. All GSSM model code and
its torch dependency live ONLY inside `external/GSSM/.venv_gssm/` (a
throwaway venv built from `/usr/bin/python3.10` + CPU-only torch==2.6.0,
never uv-synced against this project's own environment). This module
(system Python) never imports GSSM's model code or its torch build --
it only builds plain numpy feature arrays, writes them to a temp .npz,
shells out to `.venv_gssm/bin/python external/GSSM/gssm_infer.py` as a
subprocess, and reads back a numpy array of scores. No GSSM module object,
state_dict, or torch tensor of theirs ever crosses into this process.

Aggregation convention: same "worst agent, worst moment" convention as
`baselines_extra.instance_rss_score`/`ssm_baselines.instance_ssm_scores` --
the instance-level score is the MAX intensity over every (agent, frame)
in the instance (higher intensity = riskier, so max = worst-case).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import BoxTrajectory

_HERE = Path(__file__).resolve().parent
GSSM_REPO_DIR = _HERE.parent / "external" / "GSSM"
GSSM_VENV_PYTHON = GSSM_REPO_DIR / ".venv_gssm" / "bin" / "python"
GSSM_INFER_SCRIPT = GSSM_REPO_DIR / "gssm_infer.py"
GSSM_CHECKPOINT = (
    _HERE.parent / "external" / "GSSM_data" / "PosteriorInference" / "SafeBaseline"
    / "current+acc" / "model_final_71epoch.pth"
)

# Exact column order the released `current+acc` checkpoint expects
# (`src_posterior_inference/inference_utils/utils_data.py:DataOrganiser.read_data`,
# 'acc' branch).
CURRENT_ACC_COLUMNS = [
    "l_ego", "l_sur", "combined_width", "vy_ego", "vx_sur", "vy_sur",
    "v_ego2", "v_sur2", "delta_v2", "delta_v", "psi_sur", "acc_ego", "rho",
]


# --------------------------------------------------------------------------
# Rotation/angle helpers, ported verbatim (numpy, vectorized over time) from
# the official repo's `coortrans.py` -- see module docstring.
# --------------------------------------------------------------------------

def _rotate_coor(x_axis: np.ndarray, y_axis: np.ndarray, x2t: np.ndarray, y2t: np.ndarray):
    """Rotate vector (x2t, y2t) into the frame whose y-axis points along
    (x_axis, y_axis) (`coortrans.rotate_coor`, ported verbatim)."""
    norm = np.sqrt(x_axis ** 2 + y_axis ** 2)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    x = y_axis / norm * x2t - x_axis / norm * y2t
    y = x_axis / norm * x2t + y_axis / norm * y2t
    return x, y


def _angle(v1x: np.ndarray, v1y: np.ndarray, v2x: np.ndarray, v2y: np.ndarray) -> np.ndarray:
    """Signed angle from vector (v1x, v1y) to vector (v2x, v2y)
    (`coortrans.angle`, ported verbatim)."""
    sin = v1x * v2y - v2x * v1y
    cos = v1x * v2x + v1y * v2y
    return np.arctan2(sin, cos)


def gssm_current_acc_features(ego: BoxTrajectory, agent: BoxTrajectory, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame GSSM `current+acc` feature matrix (T, 13) and observed
    spacing (T,) for one (ego, agent) pair, replicating
    `ContextSegmenter.segment_data`'s per-scene feature construction
    (official repo) but evaluated at EVERY frame of the instance (not
    resampled to fixed 2.5s scenes -- GSSM's `current`/`current+acc`
    encoder is a per-frame conditional model, no history is needed for
    this configuration, so this is a faithful frame-by-frame application,
    not an approximation of their windowing).

    Deviations from the official feature pipeline, documented honestly:
    - `acc_ego` (SHRP2's EKF-filtered longitudinal acceleration in the
      original data) is approximated here as d(speed)/dt via
      `np.gradient` of the ego speed trace -- a signed tangential
      acceleration proxy, ignoring any lateral/centripetal component,
      matching this project's own convention elsewhere (e.g.
      `baselines_extra.rss_margin`'s speed-from-gradient use).
    - Velocity direction is assumed equal to heading direction (vx =
      v*cos(heading), vy = v*sin(heading)) for both ego and agent, exactly
      matching the official repo's OWN convention for all three of its
      training datasets (`transform_coor` falls back to this whenever
      explicit vx/vy columns are absent, and all three of GSSM's own
      dataset-preparation scripts construct vx/vy this way from the
      start -- i.e. this is not a simplification introduced here, it is
      the training-time assumption already baked into the released
      checkpoint). Speed magnitude itself comes from true finite
      differencing of the position trace (`np.gradient`), using this
      project's own robust per-frame `heading` field
      (`geometry.heading_from_positions`, already computed by every
      dataset adapter) rather than a raw orientation field.
    """
    T = len(ego)
    hx_e, hy_e = np.cos(ego.heading), np.sin(ego.heading)
    hx_s, hy_s = np.cos(agent.heading), np.sin(agent.heading)

    vx_e_raw, vy_e_raw = np.gradient(ego.x, dt), np.gradient(ego.y, dt)
    vx_s_raw, vy_s_raw = np.gradient(agent.x, dt), np.gradient(agent.y, dt)
    v_e = np.hypot(vx_e_raw, vy_e_raw)
    v_s = np.hypot(vx_s_raw, vy_s_raw)
    # heading-aligned velocity, matching the official repo's own convention
    vx_e, vy_e = v_e * hx_e, v_e * hy_e
    vx_s, vy_s = v_s * hx_s, v_s * hy_s

    acc_ego = np.gradient(v_e, dt)

    # -- ego view (y-axis = ego heading) --
    vx_ego_ego, vy_ego_ego = _rotate_coor(hx_e, hy_e, vx_e, vy_e)
    vx_sur_ego, vy_sur_ego = _rotate_coor(hx_e, hy_e, vx_s, vy_s)
    hx_sur_ego, hy_sur_ego = _rotate_coor(hx_e, hy_e, hx_s, hy_s)

    vy_ego_feat = np.abs(vy_ego_ego)          # index 3 ("vy_ego"), ~ ego speed
    vx_sur_feat = vx_sur_ego                  # index 4 ("vx_sur")
    vy_sur_feat = vy_sur_ego                  # index 5 ("vy_sur")

    v_ego2 = vx_e ** 2 + vy_e ** 2             # index 6, == v_e**2
    v_sur2 = vx_s ** 2 + vy_s ** 2             # index 7, == v_s**2
    delta_v2 = (vx_e - vx_s) ** 2 + (vy_e - vy_s) ** 2   # index 8
    delta_v = np.sqrt(delta_v2) * np.sign(v_ego2 - v_sur2)  # index 9

    psi_sur = _angle(np.zeros(T), np.ones(T), hx_sur_ego, hy_sur_ego)  # index 10

    # -- relative view (y-axis = ego-minus-agent relative velocity, with a
    #    fallback to ego heading if that relative velocity is exactly zero) --
    rel_x_axis = vx_e - vx_s
    rel_y_axis = vy_e - vy_s
    zero_rel = (rel_x_axis == 0) & (rel_y_axis == 0)
    rel_x_axis = np.where(zero_rel, hx_e, rel_x_axis)
    rel_y_axis = np.where(zero_rel, hy_e, rel_y_axis)

    dx = agent.x - ego.x
    dy = agent.y - ego.y
    x_sur_rel, y_sur_rel = _rotate_coor(rel_x_axis, rel_y_axis, dx, dy)
    rho = _angle(np.ones(T), np.zeros(T), x_sur_rel, y_sur_rel)  # index 12
    spacing = np.hypot(dx, dy)  # 's', center-to-center distance (rotation-invariant)

    l_ego = np.broadcast_to(ego.length, (T,)).astype(float)
    l_sur = np.broadcast_to(agent.length, (T,)).astype(float)
    combined_width = (np.broadcast_to(ego.width, (T,)) + np.broadcast_to(agent.width, (T,))) / 2.0

    features = np.stack([
        l_ego, l_sur, combined_width, vy_ego_feat, vx_sur_feat, vy_sur_feat,
        v_ego2, v_sur2, delta_v2, delta_v, psi_sur, acc_ego, rho,
    ], axis=1).astype(np.float32)

    small_eps = 1e-6
    spacing = np.clip(spacing, small_eps, None).astype(np.float32)
    return features, spacing


class GSSMUnavailableError(RuntimeError):
    """Raised when the isolated GSSM venv/checkpoint/repo is missing --
    surfaced honestly rather than silently falling back to a fabricated
    score."""


def _check_available():
    missing = [p for p in (GSSM_VENV_PYTHON, GSSM_INFER_SCRIPT, GSSM_CHECKPOINT) if not p.exists()]
    if missing:
        raise GSSMUnavailableError(
            "GSSM baseline is unavailable -- missing: " + ", ".join(str(p) for p in missing)
        )


def score_rows_via_subprocess(X: np.ndarray, s: np.ndarray, batch_tag: str = "gssm") -> np.ndarray:
    """Score a (N,13) feature matrix + (N,) spacing array via ONE subprocess
    call into the isolated `.venv_gssm` environment; returns the (N,)
    intensity array. Batched across an arbitrary number of instances/agents
    /frames by the caller for efficiency (subprocess overhead is paid once,
    not per instance)."""
    _check_available()
    with tempfile.TemporaryDirectory(prefix=f"{batch_tag}_") as tmp:
        in_path = Path(tmp) / "in.npz"
        out_path = Path(tmp) / "out.npz"
        np.savez(in_path, X=X.astype(np.float32), s=s.astype(np.float32))
        result = subprocess.run(
            [str(GSSM_VENV_PYTHON), str(GSSM_INFER_SCRIPT),
             "--checkpoint", str(GSSM_CHECKPOINT),
             "--input", str(in_path), "--output", str(out_path)],
            cwd=str(GSSM_REPO_DIR), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise GSSMUnavailableError(
                f"GSSM inference subprocess failed (code {result.returncode}):\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        out = np.load(out_path)
        return out["intensity"].astype(np.float64)


def build_gssm_instance_rows(
    ego: BoxTrajectory, agent_boxes: Dict[str, BoxTrajectory], dt: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Concatenate `gssm_current_acc_features` across every agent in one
    instance into (sum_T, 13) features and (sum_T,) spacing, ready to be
    appended to a larger cross-instance batch before a single scoring
    subprocess call. Returns None if the instance has no agents."""
    if not agent_boxes:
        return None
    feats, spacings = [], []
    for box in agent_boxes.values():
        f, s = gssm_current_acc_features(ego, box, dt)
        feats.append(f)
        spacings.append(s)
    return np.concatenate(feats, axis=0), np.concatenate(spacings, axis=0)


def instance_gssm_score_from_intensity(intensity_rows: np.ndarray) -> float:
    """Worst-case (max intensity) GSSM score across every (agent, frame) in
    the instance -- same "worst agent/moment drives the instance score"
    convention as `baselines_extra.instance_rss_score` /
    `ssm_baselines.instance_ssm_scores`."""
    if intensity_rows is None or len(intensity_rows) == 0:
        return 0.0
    return float(np.max(intensity_rows))
