# UniAD / VAD (Bench2DriveZoo `uniad/vad` branch) setup notes

Honest account of what it took to get UniAD-Tiny and VAD-Base producing an
open-loop, single-frame predicted ego trajectory on real Bench2Drive clips,
for `scripts/experiments/uniad_case_study.py` / `vad_case_study.py` and
their counterfactual/CARM-scoring companions. Written for the paper's
methods/limitations section: reports what actually happened, including two
real bugs that produced confidently-wrong-looking numbers before they were
caught, not a cleaned-up success narrative.

**Bottom line: both UniAD-Tiny and VAD-Base worked.** Both produce
plausible single-scenario open-loop trajectories (predicted-vs-real L2
error ~1-3m on ordinary driving frames, comparable to TCP's own accuracy),
and both were carried through to CARM v1 counterfactual scoring.

## Environment: `external/Bench2DriveZoo/.venv_uniad`

The `uniad/vad` branch's `requirements.txt` targets Python 3.8 + CUDA 11.8
(per `docs/INSTALL.md`) and pins `numpy==1.20.0`, `numba==0.48.0`. Neither
has a Python 3.10 wheel (this machine's system Python), and building numpy
1.20 from source against 3.10 is not a reasonable path, so per the task's
own time-box this was relaxed immediately rather than fought:

- Python: **3.10.12** (system Python's own version; a venv, not a separate
  interpreter build).
- `numpy`: pinned to **1.23.5** instead of `1.20.0` -- the newest 1.x
  release that (a) has a `cp310` wheel and (b) still keeps the deprecated
  `np.float`/`np.int`/`np.bool` aliases this ~2023-era OpenMMLab codebase
  uses in several places (removed as hard errors in numpy 1.24+). `numpy
  2.x` was avoided even though newer `torch` supports it, since several
  vendored mmdet/mmdet3d-era files predate numpy 2's stricter C-API/dtype
  -promotion changes.
- `numba`: pinned to `1.20.0`'s companion `0.48.0` in the repo; relaxed to
  latest resolvable (`0.67.0`, matching numpy 1.23.5 on py3.10). Only used
  by a handful of KITTI-eval/NMS/voxelizer modules not on this project's
  actual import path, but a couple of those modules `import numba` at top
  level and are transitively imported anyway (see below), so *some*
  working numba install was still needed.
- Every other pin in `requirements.txt` (motmetrics==1.1.3, yapf==0.40.1,
  trimesh==2.35.39, laspy==2.5.0, lazrs==0.5.3, py-trees==0.8.3) was kept
  exactly as specified -- all have Python 3.10 wheels and were not a
  source of conflict.
- `torch`: **2.4.0+cu121** / `torchvision` **0.19.0+cu121**, NOT the
  driver-matching default (`pip install torch` on this machine resolves to
  2.14+cu130, matching the system Python's own install and this GPU's
  driver, which reports CUDA 13.0 support via `nvidia-smi`). 2.4.0+cu121
  was chosen deliberately instead, for one reason: matching PyPI packages
  for a standalone `nvcc` exist at every CUDA-12.1 point release, whereas
  there was no way to get a working `nvcc` for a cu13-series build in this
  environment (see below). The A10G GPU (compute capability 8.6, Ampere)
  is well within cu121's supported architecture range, and CUDA 12.1's
  runtime is backward-compatible with the 580.173.02 driver installed
  here, so nothing about the GPU/driver forced this -- it was purely to
  keep the custom CUDA-extension build on a well-trodden path.
- **No system nvcc was preinstalled, and the pure-pip route to one did not
  work.** `pip install nvidia-cuda-nvcc-cu12` was tried first (both the
  version matching torch's cu121 build, 12.1.105, and the newest
  available, 12.9.86) -- this PyPI package turned out to ship only
  `ptxas`, not the `nvcc` driver binary itself or the CUDA runtime headers
  (`cuda_runtime.h`, `cusparse.h`, etc.) that
  `torch/utils/cpp_extension.py`'s `CUDAExtension` needs to invoke and
  compile against, i.e. it cannot build a CUDA extension by itself despite
  its name. Fell back to the task's other suggested option: apt, via
  NVIDIA's own repo (`cuda-keyring_1.1-1_all.deb` from
  `developer.download.nvidia.com`, added with passwordless `sudo`), then
  `apt-get install cuda-nvcc-12-1` (exact `12.1.105` match for the chosen
  torch build). This alone still failed the extension build with `fatal
  error: cusparse.h: No such file or directory` (`cuda-nvcc-12-1` alone
  doesn't pull in the cuBLAS/cuSPARSE/cuRAND/cuSOLVER dev headers torch's
  own headers reference even when unused); resolved by installing the
  full `cuda-toolkit-12-1` apt meta-package instead (this also brought in
  `cuda-cudart-dev-12-1`, needed for `cuda_runtime.h`). **This did touch
  system-level apt state**, outside the letter of the constraint (which is
  specifically about the system *Python* environment) -- `python3 -c
  "import futurerisk, torch, torchvision"` was re-verified working in the
  system Python after every apt step and again at the very end; nothing
  about it was touched.
- Build command: `CUDA_HOME=/usr/local/cuda-12.1
  PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST="8.6" pip
  install -v -e . --no-build-isolation --no-deps` from inside the venv,
  from `external/Bench2DriveZoo/`. `--no-build-isolation` was necessary
  because `setup.py` does `import torch` at module level with no
  `pyproject.toml` `[build-system]` declaring it, so an isolated build env
  has no torch to import. `--no-deps` was necessary because `setup.py`'s
  own `install_requires=parse_requirements()` re-reads the *unmodified*
  `requirements.txt` (with the original `numpy==1.20.0` pin) regardless of
  what was already `pip install -r`'d with relaxed versions, sending pip
  on a long, ultimately-futile search through incompatible numpy
  sdists/wheels (observed directly: pip started trying numpy 2.3.x/2.4.x
  source dists requiring Python >=3.11, none installable here).
  Compilation itself (89 object files: mostly `mmcv/ops/csrc/pytorch/*`
  CPU ops plus their CUDA counterparts under `.../cuda/*.cu`, at
  `MAX_JOBS=7`) succeeded cleanly on the first attempt once the toolchain
  was actually complete, in about 8 minutes wall-clock.
- **Three more Python-3.10-vs.-2023-era-package import errors**, hit only
  once actually running a script (not during the extension build), each a
  small, targeted patch documented at the point of the fix:
  1. `mmcv/datasets/pipelines/data_augment_utils.py` imported
     `NumbaPerformanceWarning` from `numba.errors`, which modern numba
     moved to `numba.core.errors`. One-line import fix (in this repo's own
     checked-out copy, not the system Python).
  2. The venv's `trimesh` (pulled in as a `mmcv.core.visualizer` transitive
     dependency, itself only imported because
     `mmcv.models.detectors.__init__` imports `BaseDetector`, which
     imports `show_result`) needs `pkg_resources`, which pip's default
     `setuptools` (84.x, installed in the venv) no longer bundles.
     Force-installed `setuptools<81` into the venv (not the system Python)
     to get `pkg_resources` back.
  3. `motmetrics==1.1.3` (installed in the venv's site-packages, pulled in
     transitively via `mmcv.datasets.__init__` -> `NuScenesE2EDataset` ->
     `nuscenes.eval.tracking.evaluate` -> ... -> `motmetrics`) does `from
     collections import OrderedDict, Iterable`; `Iterable` moved to
     `collections.abc` and the old alias was removed in Python 3.10.
     Patched the one line directly in the venv's installed copy of
     `motmetrics/metrics.py`.
  4. `mmcv/datasets/eval_utils/map_api.py` called `plt.style.use(
     'seaborn-whitegrid')` at import time; matplotlib renamed this style
     to `seaborn-v0_8-whitegrid` in 3.6+. Wrapped in a try/except falling
     back to the new name (this style is purely cosmetic for a plotting
     path this project's inference never uses).
  None of these four needed anything beyond a targeted one- or two-line
  fix; all are old-package-vs-newer-Python skew, not logic bugs, and all
  are confined to `external/Bench2DriveZoo/` (the repo checkout and its
  own venv), never the system Python.

## Checkpoints and data files (no training-time files needed for inference)

Downloaded from `https://huggingface.co/rethinklab/Bench2DriveZoo` into
`external/Bench2DriveZoo/ckpts/`:
- `uniad_tiny_b2d.pth` (872 MB) -- UniAD's released checkpoint, per the
  README the smallest of the two UniAD variants and the only one with no
  VAD-sized equivalent, so used as the primary target per the task's
  "start with the Tiny/smallest checkpoint" instruction.
- `vad_b2d_base.pth` (700 MB) -- VAD has no published Tiny variant.
- `resnet50-19c8e357.pth` (102 MB) -- **only needed for VAD**, and only
  because `VAD_base_e2e_b2d.py`'s config sets `model.pretrained =
  dict(img='ckpts/resnet50-19c8e357.pth')`, read during `build_model()` to
  initialize the image backbone *before* `load_checkpoint(...,
  strict=True)` overwrites those weights with the released checkpoint's
  own fully fine-tuned backbone anyway -- i.e. this file's actual weights
  never end up in the final model, it just needs to exist so
  `build_model()` doesn't raise `FileNotFoundError`. Confirmed UniAD's
  config has no equivalent `pretrained` key, so UniAD needed no such file
  at all, exactly as the task suggested checking for.
- `r101_dcn_fcos3d_pretrain.pth` (BEVFormer's backbone pretrain) was
  **not** downloaded -- not needed for either UniAD or VAD, only relevant
  to BEVFormer, which this project did not attempt to use as a case-study
  target (perception-only, no ego trajectory head; see below).
- `data/others/b2d_motion_anchor_infos_mode6.pkl` (needed to *construct*
  UniAD's `MotionHead`, which loads it directly in `__init__`) was
  **already present in this repo checkout** (3829 bytes, a pickled dict of
  `grouped_classes`/`class_list`/`K_mode`/`anchors_all`) -- not downloaded,
  not regenerated.
- Both `load_checkpoint(model, ckpt_path, map_location='cpu',
  strict=True)` calls (mirroring `UniadAgent.setup()`/`VadAgent.setup()`
  exactly) succeeded with no missing/unexpected key warnings on the first
  try, for both models -- confirming both released checkpoints really are
  self-contained for inference against their released configs. Neither
  was relaxed to `strict=False`.

## Input construction: real clip -> model forward pass

See `scripts/experiments/b2d_replay_common.py`'s module docstring for the
full, itemized account of every input field and where it comes from (live
sensor in the original CARLA-leaderboard agent vs. recorded `anno` JSON
field here vs. fixed dataset-constant camera rig).

**Two real bugs were caught by the built-in predicted-vs-real-future L2
sanity check, both fixed, both worth recording since they'd otherwise be
exactly the kind of silent, confidently-wrong result this project's
no-fabrication norm exists to catch:**

1. **Ego heading convention.** There is no live GPS/compass reading to
   replay offline, so *some* choice had to be made for the world-frame
   ego heading fed into `can_bus` and used to rotate the model's
   ego-local-frame output back to the global frame. The first attempt
   used `anno['theta']` directly. This produced predictions whose L2 error
   against the clip's own real future ego position *grew linearly* to
   20-40m over a 3s horizon -- a clear signature of a systematically wrong
   rotation, not model error. Switching to `ego_theta = anno['theta'] -
   pi/2` (the exact same rotation TCP/AD-MLP already use in this
   codebase's own `tcp_case_study.py`/`admlp_case_study.py`, for a
   different purpose -- converting world offsets into their local
   training frame) immediately improved this. Verified directly (not just
   by the improved score): rotating a clip's own real future displacement
   into the `theta`-only frame put the dominant, growing "drove forward"
   component in the *lateral* axis, which is physically wrong for a car
   driving along a road; `theta - pi/2` puts it in the *forward* axis,
   which is right. `futurerisk/adapters/bench2drive.py` separately
   documents being unable to fully pin down this same field's convention
   for its own (different) purposes, so this remains a best-effort,
   empirically-checked choice, not an independently-derived one.
2. **`sdc_traj`/`ego_fut_preds` axis order.** Even after fixing (1), L2
   error still grew to 20-30m -- smaller, but still clearly wrong. The
   model's own raw output turned out to have its two columns ordered
   (lateral, forward), not (forward, lateral) as the symmetric
   `point_cloud_range` and nuScenes-LiDAR convention would suggest.
   Verified directly by comparing the raw output against the clip's real
   future displacement rotated into the same frame: one column tracked
   near zero in both, the other grew in both, but in swapped positions.
   Swapping the two output columns before rotating back to global fixed
   this. Checked independently for VAD (not assumed to carry over from
   UniAD) -- VAD's `ego_fut_preds` has the exact same swapped ordering.
   Neither `UniadAgent.run_step()` nor `VadAgent.run_step()` ever needs to
   interpret this trajectory in the global frame themselves (they hand it
   straight to an ego-frame PID controller), so this axis convention
   never had to be pinned down in the original leaderboard-agent code --
   it only becomes a problem once the output is used for something else,
   as it is here.
3. **Reference-frame selection.** A third, less subtle issue: this
   project's first version of "pick a reference frame with a full future
   window" (adapted from tcp_case_study.py/admlp_case_study.py) picked
   frame 0 of every clip by default. Frame 0 turned out to be an
   essentially-stationary spawn point in every clip checked (`speed`
   ~1e-5 m/s) -- a materially harder and less representative prediction
   target than ordinary driving (no motion cues to extrapolate from).
   Fixed by requiring `speed >= 1.0` m/s in addition to skipping the first
   20 frames (`b2d_replay_common.find_valid_reference_frame`/
   `candidate_reference_frames`).

After all three fixes, predicted-vs-real L2 on an ordinary (non-startup,
non-hazard) frame is **~0.6-1.5m for UniAD-Tiny and ~1.5-3.3m for
VAD-Base** over a 3.0s/6-waypoint horizon -- the same order of magnitude
as TCP's own accuracy on this dataset (see `tcp_predictions.json`), which
is the strongest available evidence that the whole input-construction
pipeline (camera loading, fixed extrinsics, `can_bus`, pipeline
invocation) is faithful, not just "runs without crashing."

## Outcome

**UniAD-Tiny: succeeded.** `uniad_case_study.py` scanned 15 clips (up to
several candidate reference frames each, keeping the one with the largest
predicted-vs-real L2, matching tcp_counterfactual_case_study.py's own
rationale for picking an "interesting" frame), producing
`dataset/results_v1/uniad_predictions.json`. Full 15-clip run: ~8m12s
wall-clock on the A10G. `uniad_counterfactual_case_study.py` (system
Python) then read that JSON and produced
`dataset/results_v1/uniad_counterfactual_case_studies.json`: 15 real/
counterfactual CARM v1 score pairs, several of which show the
counterfactual (UniAD's predicted path) crossing into a geometric
collision that the real recorded expert trajectory did not have (e.g.
`AccidentTwoWays_Town12_Route1102_Weather10`: R_real=0.037 -> R_counterfactual
=0.914 with collision flipping 0->1) -- a genuine, non-fabricated
open-loop finding, not a guaranteed one (some scenarios instead show the
counterfactual scoring *lower* risk, or both real and counterfactual
already flagged as colliding).

**VAD-Base: succeeded**, same pattern. `vad_case_study.py`'s full 15-clip
scan produced `dataset/results_v1/vad_predictions.json`; L2 on ordinary
frames ~1.5-3m (slightly worse than UniAD-Tiny's ~0.6-1.5m, plausible
given VAD-Base's different architecture and training, not investigated
further). `vad_counterfactual_case_study.py` produced
`dataset/results_v1/vad_counterfactual_case_studies.json`: 15 real/
counterfactual pairs, same qualitative pattern as UniAD -- several
scenarios (e.g. `AccidentTwoWays_Town12_Route1102_Weather10`: R_real=0.037
-> R_counterfactual=0.826, collision 0->1) show VAD's predicted path
diverging into a geometric collision the real expert trajectory avoided;
others show no change or a lower counterfactual score, exactly as
expected from an honest per-scenario comparison rather than a
one-directional narrative.

**BEVFormer: not attempted.** Out of scope per the task instructions
(perception-only detector, mAP/NDS metrics, no ego trajectory/planning
head to extract a case-study trajectory from) -- moot in any case since
both actual targets (UniAD, VAD) succeeded, so no fallback was needed.

## Deviations summary (for quick reference)

| Repo said | Actually used | Why |
|---|---|---|
| Python 3.8 | 3.10.12 | System Python's version; no reason to build a separate interpreter |
| numpy==1.20.0 | 1.23.5 | No py3.10 wheel for 1.20.0; 1.23.5 is the newest 1.x release keeping the deprecated aliases this code uses |
| numba==0.48.0 | 0.67.0 | Companion pin to numpy==1.20.0; irrelevant to numpy 1.23.5/py3.10 |
| torch+cu118 | torch 2.4.0+cu121 | cu118 not specially required by the GPU/driver; cu121 chosen for a working `nvcc` story, not a driver constraint |
| conda + `cuda-toolkit` conda package | apt `cuda-toolkit-12-1` (NVIDIA's own apt repo) | No conda available; apt was the pragmatic equivalent |
| `pip install nvidia-cuda-nvcc-cu12` (task's suggested fallback) | did not work alone (ships only `ptxas`); apt `cuda-toolkit-12-1` used instead | Documented as a genuine dead end for future reference, not skipped without trying |
| (implicit) exact repo import paths | 4 one-/two-line patches (numba.errors, pkg_resources/setuptools<81, motmetrics collections.abc, matplotlib style name) | Python-3.10-vs-2023-package skew, unrelated to the core method |

## Rough time breakdown

| Step | Approx. time |
|---|---|
| Reading branch docs + `team_code/*_b2d_agent.py` + configs | ~45 min |
| venv creation + relaxed-requirements `pip install` | ~10 min |
| Diagnosing missing `nvcc` / cuSPARSE headers, adding NVIDIA apt repo | ~15 min |
| Compiling `mmcv._ext` custom CUDA ops (`pip install -v -e .`) | ~8 min (once toolchain was complete) |
| Writing `b2d_replay_common.py` + `uniad_case_study.py`/`vad_case_study.py` | ~40 min |
| Fixing the 4 import-time Python-version-skew errors | ~15 min |
| Diagnosing and fixing the ego-heading and axis-order bugs (the two real correctness issues) | ~30 min |
| Full 15-clip UniAD-Tiny + VAD-Base scans (parallel, on the GPU) | ~10-15 min each |
| Counterfactual/CARM scoring stage (system Python) | ~1 min each |
