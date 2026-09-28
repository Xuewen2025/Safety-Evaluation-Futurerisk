# GSSM baseline -- integration notes

Source: Yiru Jiao, Simeon C. Calvert, Sander van Cranenburgh, Hans van Lint,
"Learning Collision Risk Proactively from Naturalistic Driving at Scale"
(arXiv:2505.13556; Nature Machine Intelligence, 2026,
doi:10.1038/s42256-026-01189-w). Official code:
https://github.com/Yiru-Jiao/GSSM (MIT license, cloned to `external/GSSM/`).
Released checkpoints/results: https://doi.org/10.4121/9caa1e6c-9abd-4e36-ae28-c9ea4542d940
(CC BY 4.0; `PreparedData.zip`, downloaded and md5-verified against the
data repository's own `computed_md5`, then discarded after extracting the
one checkpoint used -- see below).

## Path taken: released checkpoint, zero-shot (not retrained)

The data repository's `PreparedData.zip` includes fully trained checkpoints
for every (dataset, encoder) configuration in the paper, with
`PreparedData/PosteriorInference/evaluation.csv` giving each one's
validation LogNormal-NLL. Six configurations are released for the
`SafeBaseline` (SHRP2 NDS naturalistic-driving) dataset alone, varying the
encoder: `current`, `current+acc`, `current(+acc)_environment`,
`current(+acc)_environment_profiles`.

**`SafeBaseline/current+acc/model_final_71epoch.pth`** was selected --
val_loss (LogNormal NLL) 4.590, trained on 533,345 scenes -- because it is
the only released configuration whose full input can be reconstructed from
this project's own dataset-agnostic `BoxTrajectory` fields (position,
heading, extent) with **no invented or imputed data**:

- The `environment` encoder needs SHRP2-specific categorical labels
  (lighting/weather/road-surface/traffic-density) that DeepAccident and
  Bench2Drive do not carry in any comparable form.
- The `profiles` encoder needs a 2.5s (25-frame) kinematic history window
  per scene; using it would require either inventing the first 2.5s of
  every instance's history or silently dropping instances -- not attempted.
- `current+acc` needs only 13 purely kinematic features per (ego, other
  agent, frame): vehicle lengths, combined width, ego-frame-relative
  velocity components, squared speeds and signed relative speed, relative
  heading, ego longitudinal acceleration, and `rho` (bearing of the other
  vehicle in a frame aligned with the ego-minus-agent relative-velocity
  vector -- this is the "multi-directional spacing" the paper's highlights
  refer to). All 13 are exactly reconstructible from `BoxTrajectory`.

This mirrors the project's own existing zero-shot protocol for CARM
(trained once on DeepAccident, evaluated zero-shot on Bench2Drive), except
here the "training domain" is the paper's own SHRP2 NDS release rather than
DeepAccident, and the model was never touched by anyone in this project --
same released weights, no fine-tuning, no collision label used anywhere in
producing the score.

## What the model computes, and what the risk score is

GSSM predicts a context-conditioned log-normal distribution
`log(spacing) ~ N(mu, sigma^2)` for the observed center-to-center spacing
between two road users at one frame, conditioned on the 13 kinematic
features above (via a per-feature encoder + a small self-attention
decoder). Training is label-free: the loss is exactly the log-normal
negative log-likelihood of the OBSERVED spacing given the predicted
`(mu, sigma)` -- no collision or near-crash label is used, only ordinary
naturalistic driving; risk is defined post-hoc as how atypical (close) an
observed spacing is relative to what the model learned as "normal" for
that context.

The risk score used here ("intensity") is GSSM's own function
`torch_intensity` (imported unmodified from the cloned official repo, not
re-derived): given the predicted `(mu, sigma)` and the observed spacing
`s` at a frame, it computes `log10(n)` where `n` is the number of i.i.d.
draws from the predicted distribution that would be needed for the
minimum of `n` draws to have a 50% chance of being at least as close as
`s`. Higher intensity = a deeper excursion into the close tail of "normal"
spacing for that context = riskier. This is exactly the score the paper's
own `evaluate_safety.py`/`risk_evaluation.py` use for conflict/warning
detection.

## Feature construction and documented deviations

`futurerisk/baselines_gssm.py:gssm_current_acc_features` ports the
official repo's `ContextSegmenter.segment_data` (feature definitions) and
`coortrans.rotate_coor`/`angle` (rotation helpers) verbatim into numpy,
applied at every frame of every instance (not resampled to fixed 2.5s
scenes -- the `current+acc` encoder is a per-frame conditional model with
no history term, so this is a direct per-frame application, not an
approximation of the paper's own windowing). Two honest deviations:

1. **`acc_ego`** (SHRP2's own EKF-filtered longitudinal acceleration) is
   approximated as `d(speed)/dt` via `np.gradient` of the ego speed trace
   -- a signed tangential-acceleration proxy that ignores any lateral/
   centripetal component, matching this project's existing convention
   elsewhere (e.g. `baselines_extra.rss_margin`).
2. **Velocity direction is assumed equal to heading direction**
   (`vx = v*cos(heading)`, `vy = v*sin(heading)`) for both ego and other
   agents. This is NOT a simplification introduced here -- it is the
   official repo's OWN convention for all three of its training datasets
   (their `transform_coor` derives vx/vy this way whenever explicit
   velocity columns are absent, and all three dataset-preparation scripts
   construct their velocity columns exactly this way from the start), so
   using it here keeps the input distribution consistent with what the
   released checkpoint was actually trained on. Speed magnitude itself
   comes from true finite differencing of the position trace, and heading
   from this project's own per-frame `heading` field (already computed
   robustly for both DeepAccident and Bench2Drive by their adapters).

Instance-level aggregation: the released checkpoint scores every
(agent, frame) in an instance; the instance-level GSSM score is the MAX
intensity over all of them -- the same "worst agent, worst moment"
convention `baselines_extra.instance_rss_score`/
`ssm_baselines.instance_ssm_scores` already use for RSS/PET/DRAC.

## Environment isolation

GSSM's own code needs Python 3.10 + torch==2.6.0 (its own `pyproject.toml`
pins), which need not match this project's system Python (torch
2.14+cu130). All GSSM model code and its torch build live ONLY inside
`external/GSSM/.venv_gssm/` (built from `/usr/bin/python3.10` +
CPU-only torch==2.6.0 + numpy/pandas/scipy/tqdm/scikit-learn -- NOT
uv-synced against the main environment). `futurerisk/baselines_gssm.py`
(system Python) never imports GSSM's model code: it builds plain numpy
feature arrays, writes them to a temp `.npz`, shells out ONE subprocess
call to `.venv_gssm/bin/python external/GSSM/gssm_infer.py` for the whole
evaluation set (~17.7M (agent, frame) rows across ~12k instances, scored
in one batched pass in ~77 minutes on CPU), and reads back a plain numpy
array of scores. No GSSM module object, state_dict, or torch tensor of
theirs ever crosses into the main process.

Verified after the fact: `/usr/bin/python3 -c "import futurerisk, torch;
print(torch.__version__)"` -> `2.14.0+cu130` (unchanged, system
environment not contaminated by the isolated venv).

## Results

Computed by `scripts/experiments/gssm_baseline_eval.py` on the EXACT SAME
split/instances/labels/severity-proxy computation `v1_full_eval.py` already
uses (confirmed by the finegrained pool matching root.tex's own reported
`n=6630` and `n=286` exactly).

### Table I -- instance-level discrimination vs. binary collision label

| Dataset | n | n_collisions | AUROC | AUPRC |
|---|---|---|---|---|
| DeepAccident in-domain (native label, test split) | 1656 | 320 | 0.567 | 0.226 |
| Bench2Drive cross-domain (geometric label) | 3679 | 346 | 0.817 | 0.308 |

For context (existing methods, same n, from `v1_full_eval_summary.json`):
DeepAccident in-domain AUROC/AUPRC -- CARM 0.793/0.490, y_analytic
0.648/0.372, raw 3D-TTC 0.646/0.350, RSS 0.529/0.199, Lu et al.
0.662/0.452. Bench2Drive cross-domain -- CARM 0.808/0.328, y_analytic and
raw 3D-TTC both 1.000/1.000 (geometric label near-trivially separable by
construction, already flagged elsewhere in this project), RSS 0.662/0.155,
Lu et al. 0.990/0.933.

GSSM's DeepAccident in-domain result (0.567/0.226) is weak -- only
marginally above chance and above RSS, well below every other method here.
Its Bench2Drive cross-domain result (0.817/0.308) is comparatively strong,
in fact edging out CARM's own AUROC on this split, though still short of
Lu et al.'s near-perfect (and likely overstated, given the same-label
concern above) numbers.

### Table II -- fine-grained rho vs. severity proxies, DeepAccident collision-free

| Proxy | n | rho |
|---|---|---|
| a(t) (max evasive severity) | 6630 | **+0.070** (p=1.1e-8) |
| worst-case margin m | 286 | +0.080 (p=0.18, not significant) |
| min 3D-TTC | -- | too few finite values (same as other methods' omitted column) |

For context: CARM rho(a(t))=0.276, RSS=0.363, DRAC=0.245 (all clearly
positive); y_analytic=0.007, raw 3D-TTC=0.006, Lu et al.=-0.001 (near
zero); PET=-0.093. GSSM's +0.070 is small but statistically significant
(n=6630) and sits just above the "near zero" cluster, well below CARM/RSS/
DRAC. Against the worst-case margin (n=286) the correlation (+0.080) is
not statistically distinguishable from zero.

## Honest assessment

GSSM zero-shot, on this project's DeepAccident/Bench2Drive evaluation
protocol, is a mixed-to-weak baseline: notably better than chance on
Bench2Drive cross-domain (where several methods including CARM itself do
well), but weak on DeepAccident in-domain discrimination and on
fine-grained severity ranking. This is a plausible outcome, not a bug: the
released checkpoint was trained exclusively on real-world SHRP2 NDS
naturalistic driving (US highways/arterials, human-driven, near-crash
events selected by VTTI's own criteria) and is being applied zero-shot to
CARLA-simulated, deliberately engineered crash scenarios (DeepAccident) --
arguably a larger domain gap than CARM's own DeepAccident->Bench2Drive
transfer, which at least keeps the same simulator family. No number in
this file or in `gssm_baseline_eval.json` was fabricated or adjusted; this
is what the released checkpoint actually produces on these instances.

## Time spent

Approximately 2.5 hours end-to-end (repo/paper reading and architecture
verification, locating and downloading the released checkpoint from the
data repository, isolated-venv setup, feature-pipeline port + self-check
against a synthetic two-vehicle scenario, and the ~80-minute batched
scoring run itself) -- within the requested time-box.
