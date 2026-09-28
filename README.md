# Safety Evaluation — FutureRisk / CARM

Implementation and experiments for **FutureRisk**, a post-hoc,
outcome-agnostic safety-evaluation metric for already-executed AV
trajectories, and **CARM** (Context-Adaptive Risk Model), the learned
component that produces it. The full method specification is in
[`instruction.txt`](instruction.txt); the experiment/validation plan is in
[`experiment_design.txt`](experiment_design.txt).

## Repository layout

| Path | Contents |
|---|---|
| `futurerisk/` | Core method implementation (geometry, interaction typing, safety prior, CARM model, losses, training, data pipeline, evaluation, dataset adapters) — see `futurerisk/README.md` |
| `scripts/` | Training / experiment entry points (per-dataset training, cross-domain evaluation, case studies, patches) |
| `icra/` | Paper/experiment notes and write-ups |
| `output/` | Literature-search notes and other generated artifacts |
| `dataset/` | **Small** result/checkpoint/preprocessing artifacts only (see below) |
| `external/carla-benchmark/` | Vendored copy of the TRACE benchmark (small enough to include directly) |
| `external/GSSM_data/` | GSSM posterior-inference outputs used by the GSSM baseline |

## What is *not* in this repo

The full working directory this project was developed in is **~350GB**,
almost entirely raw/preprocessed driving datasets and vendored model repos
with weights. GitHub has a 100MB per-file limit and is not meant to host
datasets at this scale, so the following are **excluded** and must be
downloaded separately if you need to reproduce a pipeline from raw data:

| Excluded path (original) | Size | What it is |
|---|---|---|
| `dataset/Bench2Drive-Base/` | ~312GB | Full Bench2Drive training clips |
| `dataset/Bench2Drive-Base-preprocessed/` | ~340MB | Preprocessed Bench2Drive-Base cache |
| `dataset/Bench2Drive-mini/` | ~2.7GB | Bench2Drive "mini" subset (raw clips) |
| `dataset/DeepAccident-motion/` | ~12GB | DeepAccident motion-data export (raw) |
| `dataset/DeepAccident-extracted/` | ~12GB | DeepAccident, extracted/decompressed |
| `dataset/DeepAccident-preprocessed/` | ~2.8GB | Preprocessed DeepAccident cache |
| `external/Bench2DriveZoo/` | ~9.9GB | Vendored UniAD/VAD codebase + checkpoints + venvs |
| `external/GSSM/` | ~1.2GB | Vendored GSSM codebase + venv (see `external/GSSM_data/` above for the small outputs that *are* included) |

What **is** kept: everything under `dataset/` that is a small result,
checkpoint, log, or preprocessed-cache directory (`results*/`,
`checkpoints/`, `logs/`, `*-preprocessed/` other than the two large ones
above, `DeepAccident-mini/`, `download.log`) — collectively a few tens of
MB, useful for inspecting past runs without needing the raw data.

## Downloading the datasets

### Bench2Drive (`Bench2Drive-Base`, `Bench2Drive-mini`)

- Project / code: <https://github.com/Thinklab-SJTU/Bench2Drive>
- Full clip data: <https://huggingface.co/datasets/rethinklab/Bench2Drive-Full>
- HD-map data: <https://huggingface.co/datasets/rethinklab/Bench2Drive-Map>
- Clips are named `ScenarioName_TownID_RouteID_WeatherID.tar.gz`; download
  and extract into `dataset/Bench2Drive-Base/` (full set) or
  `dataset/Bench2Drive-mini/` (a small subset), matching the layout the
  adapters in `futurerisk/adapters/` expect.

### DeepAccident (`DeepAccident-motion`, `DeepAccident-extracted`, `DeepAccident-preprocessed`)

- Hosted on OpenDataLab: search `OpenDataLab/DeepAccident`.
- This project used the `accident_motion_data.zip` export
  (`motion_data/all_vehicles/*.tfrecords`, 202 files, ~12GB). Extract it to
  `dataset/DeepAccident-motion/`.
- Format notes (ZLIB-compressed TFRecords using the Waymo Open Motion
  Dataset schema, namespaced per role) are documented in
  `futurerisk/adapters/deepaccident.py`'s module docstring — read that
  before writing a new loader.

### TRACE (`external/carla-benchmark/`)

Already included in full in this repo (it's small, ~30MB) — no download
needed. Upstream source, for reference or updates:
<https://github.com/NahianSalsabil/carla-benchmark> (MIT license). See
`futurerisk/adapters/trace.py`'s docstring for how the sparse scenario
JSON is turned into dense trajectories.

### GSSM baseline (`external/GSSM/`, `external/GSSM_data/`)

- Code: <https://github.com/Yiru-Jiao/GSSM> (MIT license) — clone into
  `external/GSSM/` to reproduce the environment/training path.
- Released checkpoints/results used for the zero-shot baseline:
  <https://doi.org/10.4121/9caa1e6c-9abd-4e36-ae28-c9ea4542d940> (CC BY
  4.0) — `PreparedData.zip`. This repo already includes the small derived
  outputs in `external/GSSM_data/`; see
  `dataset/results_v1/gssm_baseline_notes.md` for exactly which released
  checkpoint was used and why.

### UniAD / VAD (`external/Bench2DriveZoo/`)

- Code: <https://github.com/Thinklab-SJTU/Bench2DriveZoo.git>, `uniad/vad`
  branch — clone into `external/Bench2DriveZoo/` for the case-study /
  counterfactual scripts (`scripts/experiments/uniad_case_study.py`,
  `vad_case_study.py`). Environment setup is nontrivial (Python/CUDA/numpy
  pinning); see `dataset/results_v1/uniad_vad_setup_notes.md` for the full
  account of what was needed to get it running.

## Getting started

See `futurerisk/README.md` for module-by-module documentation of the
method implementation, dependencies, and known implementation judgment
calls (places where the spec in `instruction.txt` was ambiguous). See
`experiment_design.txt` for the overall validation plan across datasets.
