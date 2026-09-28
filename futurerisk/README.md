# FutureRisk / CARM

Implementation of the method specified in `instruction.txt`: **FutureRisk**,
a post-hoc, outcome-agnostic safety-evaluation metric for already-executed
AV trajectories, and **CARM** (Context-Adaptive Risk Model), the learned
component that produces it.

## Layout

| File | Section | Contents |
|---|---|---|
| `config.py` | — | `FutureRiskConfig`: every hyperparameter mentioned ("e.g." defaults) in one place |
| `geometry.py` | S3.2 | Oriented-box SAT overlap test, geometry-aware 3D-TTC (Eq. `3dttc`) |
| `interaction.py` | S3.3 | Interaction-type heuristic (Algorithm 1), with/without lane graph |
| `safety_prior.py` | S3.4 | Analytic safe-TTC prior (Eq. `phy-prior`), margin (Eq. `margin`), risk mapping (Eq. `phi`) |
| `event_proxy.py` | S3.5 | Kinematic event signal `a(t)`, per-(dataset, interaction-type) threshold statistics (Algorithm 2), critical-event time label (Eq. `tevt`) |
| `carm_model.py` | S3.6 | `CARM` (torch `nn.Module`): encoder (Eq. `encoder`), hazard head (Eq. `hazard`), safe-TTC head (Eq. `safettc`), composition + bounded residual (Eqs. `composition`, `bounded-residual`), agent aggregation (Eq. `agg`) |
| `losses.py` | S3.7 | `L_risk` (Eq. `l-risk`), `L_evt` with at-risk masking + class-imbalance upweighting (Eq. `l-evt`), `L_rank` (Eq. `l-rank`), total objective (Eq. `total-loss`) |
| `training.py` | S3.8 | Two-stage training orchestration (freeze/unfreeze per the algorithm) |
| `data_pipeline.py` | S3.9 | Per-dataset preprocessing, threshold fitting, caching, train/val/test split bookkeeping |
| `evaluation.py` | S3.10 | AUROC/AUPRC (overall and per-group), early-warning lead time, calibration/ECE, baseline comparison, **persistence metrics** (risk-time-integral, critical duration) |
| `visualization.py` | S3.11 | Case-study `R(t)` / margin / hazard plots |
| `adapters/bench2drive.py` | S3.9 | **Bench2Drive → `RawTrajectoryInstance` adapter** (real dataset, not synthetic) |
| `tests/test_smoke.py` | — | Synthetic end-to-end shape/range sanity check |

## Dependencies

`numpy` everywhere; `torch` for `carm_model.py`, `losses.py`, `training.py`;
`matplotlib` for `visualization.py`; `scikit-learn` optionally for
`evaluation.py` (falls back to a manual numpy AUROC/AUPRC if absent).

This sandbox initially had no package manager at all (`pip` missing); the
core method modules (everything above the adapter) were first written and
checked with only `python3 -m py_compile` plus manual equation-by-equation
tracing against `instruction.txt`. `python3-numpy`/`torch`/`matplotlib`/
`scikit-learn`/`scipy` were subsequently installed via `apt` (this
environment has passwordless sudo and network access; no user data or
credentials were touched), which made it possible to actually **run**
`tests/test_smoke.py` (passes) and, more importantly, the full Bench2Drive
adapter against the real downloaded Mini dataset end-to-end
(`build_instances_from_clip` → `data_pipeline.preprocess_dataset`) — see
`adapters/bench2drive.py`'s docstring and the note below for what that run
found and fixed. If you're reading this in a fresh environment without
these packages, install them (via `apt`, `pip`, or a venv) before relying
on anything beyond `py_compile`.

**GPU**: this machine has an NVIDIA A10G (checked via `lspci`); it had no
driver installed initially. `nvidia-headless-580` + `nvidia-utils-580`
(Ubuntu's restricted-repo packages, DKMS-built against the running
`6.8.0-1063-aws` kernel, no reboot needed since `nouveau` wasn't loaded)
plus a pip-installed CUDA build of `torch` (the apt `python3-torch` is
CPU-only, so it was removed first) got `torch.cuda.is_available()` to
`True` — verified with both `nvidia-smi` and an actual GPU matmul.
`training.Batch` gained a `.to(device)` method and
`scripts/train_bench2drive.py` now trains on `cuda` automatically when
available (falls back to CPU otherwise); checkpoints are always saved as
plain CPU tensors + plain dicts (not the `FutureRiskConfig` object itself)
so they stay portable and loadable under `torch`'s `weights_only=True`
default.

## Implementation notes / places where the spec required a judgment call

1. **`safety_prior.analytic_trajectory_risk` sign convention.** S3.4 defines
   `phi(m) = sigma(-m/T_m)` (risk decreases as margin grows — correct), but
   then glosses `R_analytic` as `max_{j,t} phi(-m^j(t))`, "equivalently
   `phi(-m)`". Taken completely literally, that double-negation would make
   `R_analytic` *increase* with a larger, safer margin, contradicting `phi`'s
   own definition. Since `phi` is monotone decreasing, `max_{j,t}
   phi(m^j(t)) == phi(min_{j,t} m^j(t)) == phi(m)` (using the `m` from Eq.
   `margin`, no extra negation) — that is what's implemented, with a
   docstring flagging the discrepancy. Worth reconciling with whoever wrote
   the method description.
2. **Evasive severity `a(t)` combines ego and agent kinematics.** S3.5.1
   says `a_lon(t)` is computed "ego and, where available, agent j"; since
   interaction type and thresholds are indexed per `(t, j)`, `a(t)` is
   implemented per agent pair as `max(severity_ego(t), severity_agent_j(t))`
   (a harsh maneuver by either party marks the interaction critical) rather
   than an ego-only signal shared across all agents.
3. **Per-bin `tau_phy` in the safe-TTC head.** Per the "critical
   implementation detail" note in S3.6.3, `tau_phy^j` is treated as a
   per-bin sequence `tau_phy^j(b_k)`; `mu_s^j(b_k)` is therefore also a
   per-bin sequence (the network only predicts one scalar correction
   `g_mu(h^j)` per agent-instance, applied uniformly on top of the
   time-varying physical prior), while `sigma_s^j` stays a single scalar per
   agent-instance as the equation defines it. The tail term
   `Pr(T_s > b_K)` in Eq. `composition` uses bin `K`'s own `(mu_s, sigma_s)`.
4. **Interaction-type priority order.** Algorithm 1's if/elif chain is
   evaluated in the literal branch order (car-following → oncoming →
   crossing → merging/lane-change → other); implemented vectorized via
   sequential overwrite so the earliest-matching branch always wins even
   when multiple conditions are geometrically true at once.
5. **Frenet-frame lateral offset** (`interaction.lateral_offset_frenet_like`)
   is a straight-line projection onto ego's instantaneous heading normal,
   not a true lane-centerline Frenet frame — a reasonable lightweight
   substitute per S3.3's own description of the heuristic as "lightweight",
   but swap in a real Frenet projection if a lane graph with centerlines is
   available in a given dataset adapter.
6. **Dataset adapters.** `data_pipeline.RawTrajectoryInstance` is the
   standardized interface each dataset must be converted into (pose/box
   traces at a common `dt`, collision flags, optional lane-change
   timestamps). `adapters/bench2drive.py` and `adapters/deepaccident.py`
   are implemented and validated against real downloaded data (see their
   own sections below); nuPlan is not — it was never acquired in this
   environment, so `datasets=("DeepAccident","nuPlan","Bench2Drive")` in
   `config.py` is aspirational for that one entry.

## A significant bug found and fixed: `v_rel_close` sign was inverted

`data_pipeline._relative_closing_speed` (the "closing-speed component along
the current relative bearing" that eq:phy-prior's `tau_phy^j(t) = t_react +
v_rel_close^j(t)/a_avail + eps_unc` is built from) had its sign backwards
from its own documented contract ("positive = closing"), from the earliest
implementation of this package through every result reported before this
fix. Found while writing `ssm_baselines.py`'s PET/DRAC unit tests, which
asserted the correct physical sign for an analogous closing-speed term and
failed against the (copied) pre-fix formula.

**Evidence** (verified by running the actual function, not by hand-deriving
signs): a synthetic, unambiguously-*closing* pair (ego stationary, agent
approaching head-on at 10 m/s) returned `v_rel_close = -10`; an
unambiguously-*receding* pair returned `+10` — exactly backwards. Both
`analytic_safe_ttc_prior`'s formula and `instruction.txt`'s own eq:phy-prior
require "positive = closing" (a higher closing speed should demand a
*larger* `tau_phy`, i.e. more margin); that half of the pipeline was
implemented correctly — the bug was isolated to this one dot-product's sign.

**Impact**: every `tau_phy` value cached before this fix was corrupted —
genuinely-closing (dangerous) pairs got an artificially small `tau_phy`
(commonly floored at the clip-range minimum, 0.2s, regardless of how fast
the true closing speed was), and genuinely safely-receding pairs got an
artificially large one. This fed `y_analytic`, `margin`, and CARM's
`SafeTTCHead` input (`tau_phy_seq`) with a systematically wrong physical
prior in *every* preprocessed instance in *every* dataset used in this
project prior to the fix. `T_evt`, `interaction_type`, and the threshold
tables were unaffected (none of them depend on `v_rel_close`).

**Fix and re-validation**: `scripts/patch_tau_phy_sign_fix.py` regenerates
only the `tau_phy` field of every already-cached `PreprocessedInstance`
(using the raw-trajectory caching added earlier — no need to re-parse
Bench2Drive-Base's 312GB of archives or re-run DeepAccident's 33-minute
threshold-fitting pass) and was verified to leave every other cached field
byte-identical. After patching and retraining, DeepAccident cross-domain
zero-shot AUROC/AUPRC went from 0.590/0.259 to **0.627/0.300** and the
mixed-training run's DeepAccident/pooled AUPRC jumped from 0.323/0.360 to
**0.407/0.431** — a substantially larger, more consistent move than the
±0.01–0.05 swings later established as this setup's single-run noise band
(see the lane-graph experiment below).

## A second bug found via the same tools: `mu_s`/`sigma_s` collapse to a saturated constant

Writing `scripts/experiments/tau_safe_vs_tau_phy.py` (S-five.2's "does the
learned safe-TTC move in sensible directions relative to the analytic
prior") to sanity-check the tau_phy fix surfaced a second, independent
problem: on the post-tau_phy-fix checkpoint, `mu_s`/`sigma_s` (`SafeTTCHead`,
eq:safettc) had collapsed to within ~0.1% of their bound
(`T_safe/tau_phy` ratio pinned at 0.500–0.571, i.e. `tanh(g_mu(h))≈-1`) for
essentially every one of 392K agent-instance rows, regardless of
interaction type or relative speed — the "context-adaptive" half of CARM's
name was not actually adaptive in practice.

**Diagnosis** (by direct inspection, not guesswork): `g_mu(h)`'s raw
pre-tanh output ranged -12.2 to -1.6 across real instances (98.5% with
`|raw|>3`, deep in tanh's flat/saturated region) — the MLP was producing
real, `h`-dependent variation, just at a magnitude that a hard-saturating
nonlinearity throws away almost entirely. Root cause: `h` (the shared
encoder output every head reads) had no normalization and a large,
uncontrolled norm (~12.3 mean on a 64-dim vector); combined with
`mu_s`/`sigma_s` getting **no direct supervision** (S3.6.3: only through
the composition) while being trained *jointly* with the much
higher-capacity 25-dim `hazard_head` in Stage 1, gradient descent had every
opportunity to push L_risk's discriminative signal through `hazard_head`
instead and no real pressure to keep these two 1-scalar-output heads
well-scaled. Confirms as mechanism, not just symptom: the *separately*
Stage-2-trained `residual_head` (same MLP+tanh pattern, same `h`, but
trained *alone* with everything else frozen — no escape route available)
did **not** show this collapse (only 10.7% with `|raw|>3`).

**Fix** (`carm_model.py`/`losses.py`/`config.py`): (1) a `LayerNorm` on the
encoder's output `h` (`InteractionEncoder.out_norm`) — reduced but did not
fix the collapse alone (88.5% still `|raw|>3` after this change alone,
since `h`'s scale being controlled doesn't stop `g_mu`'s own weights from
re-amplifying); (2) `losses.safe_ttc_saturation_penalty`, a small
(`lambda_saturation=0.01`) L2-to-zero penalty directly on `g_mu`/`g_sigma`'s
raw pre-nonlinearity outputs, Stage 1 only. Together: `tanh(g_mu)` std went
from 0.0033 (collapsed) to 0.227 (~69x), `T_safe/tau_phy` ratio now spans
0.50–1.56 (vs. 0.50–0.57 before), and downstream: Bench2Drive-Base test
AUROC/AUPRC 0.914/0.636 → **0.954/0.663**, cross-domain zero-shot
AUROC/AUPRC 0.627/0.300 → **0.637/0.322**, and cross-domain calibration ECE
0.240 → **0.147** (a 39% relative reduction — `sigma_s` was equally
collapsed toward `sigma_min`, so this also fixed how well-calibrated the
model's confidence itself is, not just its ranking). DeepAccident-only
mixed-training numbers stayed flat (not degraded) after the fix.

Still an open, honestly-reported finding: grouping the (now genuinely
varying) per-instance ratios by (interaction type, relative-speed tercile)
on DeepAccident shows every one of the 15 groups landing in a similar
"less conservative than `tau_phy`" band (-0.79 to -0.93 log2-ratio) — the
specific hoped-for pattern from S-five.2 ("oncoming should demand more
margin than car-following") hasn't clearly emerged at that coarse grouping
yet, even though real per-instance variation is now present (individual
ratios span 0.50–1.59). Since `mu_s` still has no direct target, there's no
guarantee it organizes itself along *this particular* axis rather than some
other cue the network found more useful — worth a follow-up look.

## Extensions beyond the base spec

Two capabilities added on top of `instruction.txt`'s method description
(both opt-in — nothing about the base pipeline changes if you ignore them):

### A. Beyond collision rate: persistence + per-group reporting

The base metric already gives a continuous `R` instead of a binary
collision label, but a single scalar per instance still can't distinguish
"a brief, deep close call" from "a long, shallow one" — S3.1's own stated
motivation ("how closely **and how persistently**") has a "persistently"
half that wasn't exposed as a standalone number. `evaluation.py` now adds:

- `risk_time_integral(r_of_t, times)` — trapezoidal integral of `r(t)`,
  i.e. total risk *exposure* in risk-seconds (two trajectories with the
  same peak risk can have very different integrals).
- `critical_duration(margin_t, times, margin_threshold=0.0)` — total time
  spent with margin at or below a threshold (default: actually unsafe,
  `m<=0`).
- `persistence_metrics(...)` — one-call bundle of peak/mean/integral/
  duration/worst-margin for a single instance, meant to sit alongside `R`
  in a report rather than replace it.
- `group_discrimination_metrics(...)` / `persistence_summary_by_group(...)`
  — break AUROC/AUPRC and the persistence metrics out by group (dominant
  interaction type, agent class, dataset...) instead of one pooled number,
  so e.g. "discriminates well on car-following, poorly on crossing" is
  visible. Small groups are still reported but flagged
  (`low_sample_warning`) rather than silently trusted.

The "dominant interaction type per instance/agent" grouping key comes from
a new `interaction.dominant_interaction_type(types)` (mode over `t`) —
S3.6.1 already anticipated this ("interaction type ... one-hot,
time-varying **or a per-instance dominant type**"), it just wasn't
implemented anywhere yet. `data_pipeline.preprocess_instance` now computes
and caches it per agent (`PreprocessedInstance.dominant_interaction_type`).

### B. Object-class-aware `tau_phy` (context/object-adaptive safety values)

Previously every agent used the same `(a_avail, t_react, eps_unc,
tau_phy_clip)` constants in the analytic safe-TTC prior (Eq. `phy-prior`)
regardless of whether it was a car, a truck, a cyclist, or a pedestrian —
physically wrong, since "available deceleration" and reaction time aren't
comparable across those. `config.py` now has:

- `PhysicalPriorParams` — a small dataclass holding the four constants.
- `FutureRiskConfig.physical_prior_by_class` — a `{class_name:
  PhysicalPriorParams}` table with distinct, physically-motivated defaults
  for `vehicle` / `truck` / `bus` / `cyclist` / `pedestrian`, plus an
  `other` fallback bucket for unclassified objects (deliberately the *most
  conservative* profile, not vehicle-like, so an unrecognized object
  doesn't get an artificially generous safety margin).
- `FutureRiskConfig.get_physical_prior_params(agent_class)` — case-
  insensitive lookup with that fallback.
- `safety_prior.analytic_safe_ttc_prior_for_class(v_rel_close, agent_class,
  cfg)` — convenience wrapper combining the lookup and the existing
  (unchanged) `analytic_safe_ttc_prior` formula.

`data_pipeline.RawTrajectoryInstance` gained an `agent_classes: Dict[str,
str]` field (agent id → class name); `preprocess_instance` now looks up
each agent's own class before computing its `tau_phy`, instead of using
one global constant for every agent. This makes the **deterministic**
prior itself object-aware, on top of CARM's already-learned per-agent
correction (`g_mu`/`g_sigma`, S3.6.3) — the network's bounded `[0.5x, 2x]`
adjustment now only has to fine-tune around a per-class-sensible starting
point rather than fight a systematically wrong one for e.g. pedestrians.

## Bench2Drive adapter (`adapters/bench2drive.py`)

Turns the raw Bench2Drive clip archives (`.tar.gz`, one per scenario clip)
into `RawTrajectoryInstance`s that feed straight into `data_pipeline`
unchanged. The dataset's own docs don't specify several things this needed
(frame rate, which position/orientation fields are trustworthy, half- vs.
full-extent convention, how to detect a collision at all) — every one of
these was **verified empirically against the actual downloaded Mini
dataset** rather than assumed; see the module's docstring for the full
list with evidence. Highlights:

- **10Hz** confirmed by cross-checking consecutive-frame displacement
  against reported speed (lands on 0.0998–0.1004s consistently); the
  frame's top-level `x`/`y` turned out to carry simulated localization
  noise and don't match speed at all — `bounding_boxes[*]['center']` (exact
  simulator ground truth) is used instead, for every agent including ego.
- **No usable orientation field** — neither the per-box `rotation` nor the
  top-level ego `theta` could be reconciled with a verifiable, consistent
  convention (worked through in detail in `geometry.heading_from_positions`'s
  docstring) — instead heading is derived identically for every agent from
  consecutive box centers, added as a new reusable utility in `geometry.py`.
- **No collision flag anywhere in the dataset** — detected geometrically
  instead, reusing this package's own `geometry.boxes_overlap` (same-frame,
  delta=0) rather than inventing separate logic.
- **Windowing**: a raw ~20s clip isn't one "trajectory instance" in S3.1's
  sense (fixed horizon `H`, one shared ego trajectory, every agent observed
  over that *same* window) — the adapter slides a fixed `H`-second window
  over each clip, including an agent only if it's contiguously tracked for
  the window's *entire* length, which is what lets `data_pipeline` assume
  one shared time axis for ego + every agent.
- **A real bug found and fixed via actually running this against the
  downloaded data**: CARLA recycles actor ids within one recorded episode.
  A pedestrian id was observed parked off-map (z≈-100m) for ~2s, then
  instantly "teleported" ~67m to a real spawn position under the *same* id
  one frame later, because an unrelated pedestrian later reused it.
  Naively trusting "same id ⇒ same physical track" would have spliced two
  unrelated actors together and injected a ~700m/s spurious velocity spike
  at the splice, corrupting heading/evasive-severity/interaction-type right
  there. `_segment_agent_positions` now also breaks a track wherever
  consecutive frames imply an unphysical speed (>40 m/s), not just where a
  frame is missing.

**Validated end-to-end against the real download**: this environment had
no numpy/torch initially (see Dependencies above); after installing them,
`build_instances_from_directory` + `data_pipeline.preprocess_dataset` were
run against all 10 Bench2Drive-Mini clips — 34 windowed instances built,
threshold table fit successfully for all 5 interaction types (none needed
the `N_min` merge-into-"other" fallback), 166 agent-instance pairs got a
finite `T_evt` (collision or harsh-evasive proxy event) out of a few
hundred, and every cached `.npz` round-trips through
`load_preprocessed_instance` correctly. A handful of geometrically-detected
"collisions" (e.g. in the `AccidentTwoWays` scenario, ego stopped for
seconds a few meters from an already-stalled vehicle) were manually
spot-checked by hand-deriving the SAT overlap test's separating axes for
one case and confirming they're mathematically correct given the recorded
positions/headings, not an artifact — plausible given those scenarios are
specifically about navigating around stalled/crashed vehicles at close
range.

## DeepAccident adapter (`adapters/deepaccident.py`)

Turns the OpenDataLab `DeepAccident` motion-data export
(`motion_data/all_vehicles/*.tfrecords`) into `RawTrajectoryInstance`s.
Every quirk below was found by inspecting the actual downloaded files, not
assumed from the paper — see the module docstring for the full list.
Highlights:

- **ZLIB-compressed, WOMD-schema tf.Example records**, namespaced under 4
  role prefixes (`ego_vehicle`/`ego_vehicle_behind`/`other_vehicle`/
  `other_vehicle_behind`) instead of Waymo's flat numeric-index layout —
  each role's block is a complete, independent Waymo-Motion-style scenario
  centered on that role as "self".
- **Real heading given directly** (`bbox_yaw`), unlike Bench2Drive — no
  need for `geometry.heading_from_positions` here.
- **Native ground-truth collision annotation**
  (`collision_intensity`/`collision_ego_vehicle_id`/
  `collision_other_vehicle_id`) exists and is used as the *independent*
  label for cross-domain evaluation (scoring a tau_3D-derived baseline
  against a tau_3D-derived label would be circular) — and, with a
  distance-plausibility guard (a real touch/graze lands within roughly the
  two boxes' combined half-diagonal, not tens of meters away — an
  unguarded first version of this fix was verified to sometimes patch in a
  "collision" at 16–35m separation and made results measurably *worse*),
  is also used to patch `agent_collision_flags` at the training-label
  level: this package's own geometric SAT-overlap detector alone misses a
  real native collision on this dataset roughly 69% of the time (10Hz
  discretization + box-dimension noise), which was silently teaching the
  model "no collision here" on instances DeepAccident's own physics engine
  says genuinely crashed.
- **Lane graph now parsed (previously hardcoded absent).** The raw export
  actually carries a real WOMD-style point-sampled lane graph
  (`{role}/roadgraph_samples/{xyz,type,id,valid}`), but `lane_graph_available`
  was hardcoded `False` here until this was added — meaning
  `classify_interaction_type` always ran S3.3's no-lane-graph fallback, so
  the **MERGING interaction type was structurally unreachable** on this
  dataset (its no-lane-graph branch never returns it). DeepAccident's type
  codes don't necessarily match the public Waymo enum one for one, so which
  code means "lane centerline" was confirmed empirically rather than
  assumed: real ego positions sit a median 0.14m from the nearest `type==1`
  point vs. ~4.3m for `type==2` and ~2.2m for `type==20`. Per frame, the
  nearest lane-centerline point's `id` (via a per-instance
  `scipy.spatial.cKDTree`) is treated as "which lane this agent is
  currently in"; an id change between consecutive frames is the
  lane-change/merge-event proxy, mirroring Bench2Drive's
  `(road_id, lane_id)`-tuple-change proxy but built from point-cloud
  geometry. Regression test: `tests/test_deepaccident_adapter.py`.
- **`-inf` box dimensions** on some valid-position frames (~50% of
  pedestrian slots observed) — handled by taking the median of only the
  finite, positive dimension samples in an agent's used window, falling
  back to a class-typical default if none are finite.
- **`record_stride`** keeps only every Nth record per file — consecutive
  records mostly share one `scenario_id` (time-shifted snapshots of the
  same underlying incident), so this trades bounded redundancy loss for a
  proportional cut in instance count (set to 10 in
  `scripts/preprocess_deepaccident.py`, landing in the same order of
  magnitude as Bench2Drive-Base for a fair mixed-training split — an
  unthrottled `record_stride=1` run produced 21x more instances and never
  finished).
