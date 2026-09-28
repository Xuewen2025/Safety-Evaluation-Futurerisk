"""DeepAccident -> RawTrajectoryInstance adapter.

Format notes, verified against the actual downloaded
`OpenDataLab/DeepAccident` motion-data export (`accident_motion_data.zip`,
`motion_data/all_vehicles/*.tfrecords`, 202 files, ~12GB) rather than
assumed from the paper:

- Each `.tfrecords` file is **ZLIB-compressed** (raw zlib stream, magic
  bytes `78 9c`) -- must open with
  `tf.data.TFRecordDataset(path, compression_type="ZLIB")`, not the
  default. Feeding the raw file to a plain `TFRecordDataset` fails with
  "corrupted record" (not documented anywhere; found by inspecting the raw
  file header).
- Each record is a `tf.train.Example` using the **Waymo Open Motion
  Dataset's exact tf.Example schema** (`state/past/x`, `state/current/x`,
  `state/future/x`, `state/id`, `state/type`, `state/is_sdc`,
  `roadgraph_samples/*`, `traffic_light_state/*`, ...), just namespaced
  under 4 role prefixes instead of Waymo's flat numeric-index layout --
  `ego_vehicle`, `ego_vehicle_behind`, `other_vehicle`,
  `other_vehicle_behind` (matches the paper's "two vehicles designed to
  collide, two more following"). Each role's block is a **complete,
  independent Waymo-style scenario** centered on that vehicle as "self"
  (`state/is_sdc == 1` marks which of the (typically 128) tracked agent
  slots is that role's own vehicle).
- Per-agent state arrays are flattened **agent-major**: `state/past/x` has
  length `n_agents * n_past` (observed 128*10), `state/current/x` has
  length `n_agents * 1`, `state/future/x` has length `n_agents * n_future`
  (observed 128*30) -- reshape each to `(n_agents, n_steps)` and
  concatenate along time to get one (n_agents, n_past+1+n_future) block per
  field. Frame spacing is confirmed 0.1s (`state/past/timestamp_micros`
  steps by exactly 100000); total span is ~4.1s (1.0s past+current, 3.0s
  future) -- shorter than Bench2Drive's 5s windows, left as-is (the shared
  `cfg.num_bins`/`cfg.horizon_s` bin grid and existing padding logic in
  `features.py` already handle a shorter-than-`horizon_s` instance
  correctly, which matters because the SAME trained CARM checkpoint/config
  must apply to both datasets for a cross-domain comparison to mean
  anything).
- `bbox_yaw` is given directly (radians, world frame) -- unlike
  Bench2Drive, no need for `geometry.heading_from_positions`; DeepAccident
  actually provides real heading.
- `state/type`: Waymo's standard codes (1=vehicle, 2=pedestrian,
  3=cyclist, 4=other, -1=unused slot) -- confirmed empirically by checking
  typical width/length per type (type 1 ~1.98m x 4.48m = car-sized; type 3
  ~0.29m x 2.21m = bike-sized).
- Per-step `width`/`length`/`height` can be **exactly `-inf`** for some
  valid-position frames of some agents (observed ~50% of pedestrian slots'
  dimension entries) -- a real data quirk, not a parsing bug (confirmed by
  reading raw floats). Handled by taking the median of only the finite,
  positive dimension samples within an agent's used window, falling back
  to a class-typical default if literally none are finite.
- `collision_intensity`/`collision_ego_vehicle_id`/
  `collision_other_vehicle_id` are scalar **per-record** ground-truth
  collision annotations (not derived by us). Every sampled record across
  10 randomly chosen files had `collision_intensity > 0` -- this specific
  export (`accident_motion_data.zip`) appears to contain **only accident
  scenarios** (the separate "normal"/non-accident scenario data mentioned
  in the project's docs is not part of this particular file). This
  adapter does not currently consume these native fields for
  `agent_collision_flags` (keeps using this package's own geometric
  overlap test, `geometry.boxes_overlap`, for consistency with the
  Bench2Drive adapter and the rest of the pipeline) -- they would be a
  good addition for cross-validating the geometric collision detector
  against DeepAccident's ground truth in a follow-up.
- `records` within one file are grouped by `scenario_id` (several
  consecutive records typically share one `scenario_id`, evidently
  different time-offset samples of the same underlying ~10-20s scenario
  recording, analogous in spirit to Bench2Drive's sliding windows -- except
  here each record is already a complete, self-contained ~4.1s window, so
  no additional windowing logic is needed).
- `{role}/roadgraph_samples/{xyz,type,id,valid,dir}` carries a real WOMD-
  style point-sampled lane graph, namespaced per role exactly like the
  agent-state fields -- but it was NOT being parsed at all until this was
  added (`lane_graph_available` was hardcoded False, unlike Bench2Drive's
  `True`), which meant `classify_interaction_type` always ran S3.3's
  no-lane-graph fallback (no merging/lane-change branch) on this dataset.
  DeepAccident's own type-code enum doesn't necessarily match the public
  Waymo Motion Dataset's documented `roadgraph_samples/type` values one for
  one, so which type is "lane centerline" was verified empirically rather
  than assumed: across a sample of records, real ego positions sit a
  median 0.14m from the nearest `type==1` point vs. ~4.3m for `type==2`
  and ~2.2m for `type==20` -- i.e. `type==1` is the polyline vehicles
  actually drive along here, used as the lane-centerline set (falling back
  to `type in {1,2}` on the rare scene with no type-1 points at all). Per
  frame, the nearest lane-centerline point's `id` is looked up (via a
  per-instance `scipy.spatial.cKDTree`, since each record's roadgraph is
  its own small local point cloud) and treated as "which lane this
  agent/ego is currently in"; a change in that nearest id between
  consecutive frames is the lane-change/merge-event proxy, mirroring
  Bench2Drive's `(road_id, lane_id)`-tuple-change proxy (see
  `bench2drive.py`'s `_lane_change_event_times`) but built from point-cloud
  geometry instead of discrete per-box tags.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from ..config import DEFAULT_CONFIG, FutureRiskConfig
from ..data_pipeline import RawTrajectoryInstance
from ..geometry import BoxTrajectory, boxes_overlap

DATASET_NAME = "DeepAccident"
ROLES: Tuple[str, ...] = ("ego_vehicle", "ego_vehicle_behind", "other_vehicle", "other_vehicle_behind")

# A track shorter than this (at 0.1s frames, 10 frames = 1s) carries too
# little signal, same threshold rationale as the Bench2Drive adapter.
MIN_TRACK_FRAMES = 10

_TYPE_TO_CLASS = {1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
# Fallback box dimensions (length, width, height) when an agent's window
# has literally no finite dimension sample (see module docstring's -inf note).
_DEFAULT_DIMS = {
    "vehicle": (4.5, 1.85, 1.5),
    "pedestrian": (0.5, 0.5, 1.7),
    "cyclist": (1.8, 0.5, 1.7),
    "other": (2.0, 2.0, 2.0),
}


def _floats(feat, key: str) -> np.ndarray:
    return np.asarray(feat[key].float_list.value, dtype=np.float64)


def _ints(feat, key: str) -> np.ndarray:
    return np.asarray(feat[key].int64_list.value, dtype=np.int64)


def _stack_time_field(feat, prefix: str, name: str, n_agents: int, n_past: int, n_future: int,
                       as_int: bool = False) -> np.ndarray:
    """(n_agents, n_past+1+n_future), concatenating past+current+future in
    time order for one field -- the core Waymo-schema reshape."""
    getter = _ints if as_int else _floats
    past = getter(feat, f"{prefix}/past/{name}").reshape(n_agents, n_past)
    cur = getter(feat, f"{prefix}/current/{name}").reshape(n_agents, 1)
    fut = getter(feat, f"{prefix}/future/{name}").reshape(n_agents, n_future)
    return np.concatenate([past, cur, fut], axis=1)


def _longest_valid_run(valid_row: np.ndarray) -> Tuple[int, int]:
    """(start, end) half-open slice of the longest contiguous True run in
    a 1D boolean array (same pattern as the Bench2Drive adapter's
    contiguous-run helpers, reimplemented locally to keep adapters
    independent of each other)."""
    best = (0, 0)
    run_start: Optional[int] = None
    values = valid_row.tolist()
    values.append(False)  # sentinel to flush a run ending at the last real index
    for i, v in enumerate(values):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    return best


def _agent_class_for(type_code) -> str:
    return _TYPE_TO_CLASS.get(int(type_code), "other")


def _representative_dims(length_row: np.ndarray, width_row: np.ndarray, height_row: np.ndarray,
                          window: slice, agent_class: str) -> Tuple[float, float, float]:
    """Median of the finite, positive dimension samples within `window`
    (a rigid body's box shouldn't change size over time; the median is
    robust to the occasional -inf sample described in the module
    docstring). Falls back to a class-typical default if none are finite."""
    l, w, h = length_row[window], width_row[window], height_row[window]
    finite = np.isfinite(l) & np.isfinite(w) & np.isfinite(h) & (l > 0) & (w > 0) & (h > 0)
    if finite.any():
        return float(np.median(l[finite])), float(np.median(w[finite])), float(np.median(h[finite]))
    return _DEFAULT_DIMS.get(agent_class, _DEFAULT_DIMS["other"])


_LANE_CENTER_TYPE_PRIMARY = 1
_LANE_CENTER_TYPES_FALLBACK = (1, 2)


def _parse_lane_centerline(feat: dict, role: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract (xy, id) for this role's roadgraph_samples points classified
    as lane-centerline (see module docstring for how `type==1` was
    empirically confirmed to be the lane-centerline code in this export).
    Returns None if there aren't at least 2 such points (can't build a
    meaningful nearest-neighbor lookup, and/or this record's roadgraph
    export is degenerate/empty -- graceful per-instance fallback to
    lane_graph_available=False rather than raising)."""
    xyz = _floats(feat, f"{role}/roadgraph_samples/xyz").reshape(-1, 3)
    typ = _ints(feat, f"{role}/roadgraph_samples/type")
    valid = _ints(feat, f"{role}/roadgraph_samples/valid").astype(bool)
    ids = _ints(feat, f"{role}/roadgraph_samples/id")
    mask = valid & (typ == _LANE_CENTER_TYPE_PRIMARY)
    if mask.sum() < 2:
        mask = valid & np.isin(typ, _LANE_CENTER_TYPES_FALLBACK)
    if mask.sum() < 2:
        return None
    return xyz[mask, :2], ids[mask]


def _nearest_lane_ids(xy_query: np.ndarray, tree: cKDTree, lane_ids: np.ndarray) -> np.ndarray:
    """Nearest lane-centerline point's id at each query position -- the
    "which lane am I in" proxy (module docstring)."""
    _, nn_idx = tree.query(xy_query)
    return lane_ids[nn_idx]


_LANE_ID_DEBOUNCE_MIN_RUN = 5  # frames (0.5s at cfg.dt=0.1s)


def _debounce_id_sequence(seq: np.ndarray, min_run: int = _LANE_ID_DEBOUNCE_MIN_RUN) -> np.ndarray:
    """Suppress short-lived flickers in a discrete nearest-id sequence.

    Verified necessary, not a defensive guess: near DeepAccident's
    intersection-heavy accident scenarios, adjacent lane centerlines can
    sit close enough that per-frame position noise flips the nearest-id
    lookup back and forth for a frame or two -- measured directly on a
    10-file sample, 60% (332/550) of raw "lane change" events were two
    id-changes within 0.3s of each other (i.e. A->B->A), not a real,
    sustained lane change. Left unfiltered, this fed a mostly-spurious
    merging/lane-change signal into training and measurably *hurt*
    downstream AUROC/AUPRC in a full mixed-training run rather than
    helping (an instance of the same "unguarded fix made things worse"
    lesson as this module's native-collision distance guard, above).

    A run of `seq` shorter than `min_run` consecutive frames is treated as
    noise and folded back into whatever value preceded it, rather than
    accepted as a real state change; only a run at least `min_run` long
    updates the accepted "current lane" state. Single left-to-right pass
    (a good-enough heuristic, same spirit as this codebase's other
    "lightweight, not a full HD-map/behavior-planner" proxies)."""
    seq = np.asarray(seq)
    n = len(seq)
    if n == 0:
        return seq
    out = np.empty_like(seq)
    accepted = seq[0]
    i = 0
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            accepted = seq[i]
        out[i:j] = accepted
        i = j
    return out


def _lane_change_times_from_ids(lane_id_seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Timestamps where the nearest-lane id changes between consecutive
    frames -- same role as Bench2Drive's `_lane_change_event_times`, just
    keyed on a point-cloud nearest-id instead of a (road_id, lane_id)
    tuple. `lane_id_seq` is debounced first (see `_debounce_id_sequence`)."""
    lane_id_seq = _debounce_id_sequence(lane_id_seq)
    out = []
    prev = None
    for lid, t in zip(lane_id_seq, times):
        if prev is not None and lid != prev:
            out.append(t)
        prev = lid
    return np.array(out, dtype=float)


def load_records(tfrecord_path: str) -> Iterator[dict]:
    """Yield each record's `feature` map (str -> tf.train.Feature) from one
    ZLIB-compressed DeepAccident motion .tfrecords file."""
    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="ZLIB")
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        yield ex.features.feature


def _build_role_instance(
    feat: dict, role: str, clip_name: str, record_idx: int, cfg: FutureRiskConfig,
) -> Optional[RawTrajectoryInstance]:
    prefix = f"{role}/state"
    is_sdc = _ints(feat, f"{prefix}/is_sdc")
    n_agents = len(is_sdc)
    if n_agents == 0:
        return None
    sdc_candidates = np.where(is_sdc == 1)[0]
    if len(sdc_candidates) == 0:
        return None
    sdc_idx = int(sdc_candidates[0])

    n_past = len(_floats(feat, f"{prefix}/past/x")) // n_agents
    n_future = len(_floats(feat, f"{prefix}/future/x")) // n_agents

    x = _stack_time_field(feat, prefix, "x", n_agents, n_past, n_future)
    y = _stack_time_field(feat, prefix, "y", n_agents, n_past, n_future)
    z = _stack_time_field(feat, prefix, "z", n_agents, n_past, n_future)
    yaw = _stack_time_field(feat, prefix, "bbox_yaw", n_agents, n_past, n_future)
    length = _stack_time_field(feat, prefix, "length", n_agents, n_past, n_future)
    width = _stack_time_field(feat, prefix, "width", n_agents, n_past, n_future)
    height = _stack_time_field(feat, prefix, "height", n_agents, n_past, n_future)
    valid = _stack_time_field(feat, prefix, "valid", n_agents, n_past, n_future, as_int=True).astype(bool)
    obj_type = _ints(feat, f"{prefix}/type")     # (n_agents,), constant per agent
    obj_id = _ints(feat, f"{prefix}/id")          # (n_agents,), constant per agent

    T = x.shape[1]

    ego_start, ego_end = _longest_valid_run(valid[sdc_idx])
    if ego_end - ego_start < MIN_TRACK_FRAMES:
        return None

    def make_box(agent_idx: int, start: int, end: int, agent_class: str) -> BoxTrajectory:
        l, w, h = _representative_dims(length[agent_idx], width[agent_idx], height[agent_idx],
                                        slice(start, end), agent_class)
        return BoxTrajectory(
            x=x[agent_idx, start:end], y=y[agent_idx, start:end], z=z[agent_idx, start:end],
            heading=yaw[agent_idx, start:end], length=l, width=w, height=h,
        )

    ego_class = _agent_class_for(obj_type[sdc_idx])
    ego_box = make_box(sdc_idx, ego_start, ego_end, ego_class)
    times = np.arange(T) * cfg.dt
    ego_times = times[ego_start:ego_end] - times[ego_start]

    # Lane graph (S3.3's merging/lane-change branch) -- see module
    # docstring. Built once per record/role since each is its own local
    # roadgraph point cloud; falls back to lane_graph_available=False for
    # this instance alone if the roadgraph export is degenerate.
    lane_graph_available = False
    lane_tree: Optional[cKDTree] = None
    lane_ids_arr: Optional[np.ndarray] = None
    ego_lane_change_times: Optional[np.ndarray] = None
    lane_data = _parse_lane_centerline(feat, role)
    if lane_data is not None:
        lane_xy, lane_ids_arr = lane_data
        lane_tree = cKDTree(lane_xy)
        ego_lane_seq = _nearest_lane_ids(np.stack([ego_box.x, ego_box.y], axis=1), lane_tree, lane_ids_arr)
        ego_lane_change_times = _lane_change_times_from_ids(ego_lane_seq, ego_times)
        lane_graph_available = True

    # Native ground-truth collision annotation (S3.10 needs an independent
    # collision label -- deriving it from our own tau_3D geometric overlap
    # test, like the Bench2Drive adapter does, would make y_analytic/raw-TTC
    # baseline comparisons tautological with the label they're being scored
    # against, since both are functions of the same tau_3D quantity).
    # collision_ego_vehicle_id/collision_other_vehicle_id/collision_intensity
    # are scalar per RECORD (the whole scenario snippet), naming the two of
    # the (up to) 4 role-vehicles that actually collided -- so a given
    # role's own vehicle only really "collided" if its own object id is one
    # of those two AND intensity is nonzero (a zero-intensity scenario would
    # mean no collision at all, not observed in this export, see module
    # docstring, but checked defensively anyway).
    own_id = int(obj_id[sdc_idx])
    collision_intensity = float(_floats(feat, "collision_intensity")[0])
    collision_ego_id = int(_ints(feat, "collision_ego_vehicle_id")[0])
    collision_other_id = int(_ints(feat, "collision_other_vehicle_id")[0])
    native_is_collision = collision_intensity > 0 and own_id in (collision_ego_id, collision_other_id)
    partner_id = None
    if native_is_collision:
        partner_id = collision_other_id if own_id == collision_ego_id else collision_ego_id

    agent_boxes: Dict[str, BoxTrajectory] = {}
    agent_classes: Dict[str, str] = {}
    agent_collision_flags: Dict[str, np.ndarray] = {}
    agent_lane_change_times: Dict[str, np.ndarray] = {}
    idx_id = np.arange(ego_end - ego_start)

    for agent_idx in range(n_agents):
        if agent_idx == sdc_idx or obj_type[agent_idx] < 0:
            continue
        a_start, a_end = _longest_valid_run(valid[agent_idx])
        if a_start > ego_start or a_end < ego_end:
            continue  # this agent's own track doesn't cover ego's whole window
        agent_class = _agent_class_for(obj_type[agent_idx])
        agent_box = make_box(agent_idx, ego_start, ego_end, agent_class)
        aid = str(int(obj_id[agent_idx]))
        agent_boxes[aid] = agent_box
        agent_classes[aid] = agent_class
        agent_collision_flags[aid] = boxes_overlap(ego_box, idx_id, agent_box, idx_id, check_vertical=True)
        # agent_box's window is always sliced to exactly [ego_start:ego_end]
        # (guarded above), so it shares ego_times' time axis 1:1 -- no
        # separate re-slicing needed for the lane-id lookup.
        if lane_graph_available:
            agent_lane_seq = _nearest_lane_ids(np.stack([agent_box.x, agent_box.y], axis=1), lane_tree, lane_ids_arr)
            agent_lane_change_times[aid] = _lane_change_times_from_ids(agent_lane_seq, ego_times)

    if not agent_boxes:
        return None

    # Patch in the native ground truth: our own SAT overlap test misses a
    # real collision ~69% of the time on this dataset (confirmed by
    # comparing against this same native label -- likely 10Hz frame
    # discretization missing the exact impact instant, plus box-dimension
    # noise, e.g. the -inf sentinel values described in the module
    # docstring). Training's T_evt (eq:tevt) reads collision(t) straight
    # from `agent_collision_flags`, so leaving those 69% at all-False
    # silently taught the model "no collision here" on instances that
    # DeepAccident's own physics engine says genuinely crashed -- corrupting
    # exactly the supervision signal this dataset was added to strengthen.
    # Where the known partner IS one of this instance's tracked agents, we
    # force its flag True at the frame of closest ego-partner approach (the
    # best time estimate obtainable from this instance's own trajectory
    # data, since the native annotation carries no per-frame timestamp).
    #
    # Plausibility guard -- REQUIRED, not optional: `native_is_collision` is
    # constant across every consecutive record sharing one `scenario_id`
    # (several time-shifted ~4s windows into the same underlying incident,
    # see this module's docstring), but only ONE of those windows actually
    # contains the real impact instant. An earlier version of this fix
    # forced the flag at this window's closest approach regardless of how
    # far that still was, and verified directly to sometimes be doing so at
    # 16-35m separation (confirmed by inspecting real flagged instances) --
    # fabricating a "collision" in windows that never saw contact at all,
    # measurably *worse* than leaving the gap alone (both DeepAccident's own
    # and, through the shared encoder, Bench2Drive's held-out AUROC/AUPRC
    # all dropped after training on the unguarded version). Only accept the
    # patch if the closest approach is within a generous multiple of the two
    # boxes' own combined half-diagonal -- a real touch/graze lands at
    # roughly that geometric scale, not tens of meters away.
    partner_key = str(partner_id) if partner_id is not None else None
    native_fix_applied = False
    if partner_key is not None and partner_key in agent_boxes:
        partner_box = agent_boxes[partner_key]
        dist = np.hypot(partner_box.x - ego_box.x, partner_box.y - ego_box.y)
        closest_frame = int(np.argmin(dist))
        closest_dist = float(dist[closest_frame])
        ego_half_diag = 0.5 * np.hypot(ego_box.length[closest_frame], ego_box.width[closest_frame])
        partner_half_diag = 0.5 * np.hypot(partner_box.length[closest_frame], partner_box.width[closest_frame])
        plausible_contact_dist = (ego_half_diag + partner_half_diag) * 2.0  # generous slack
        if closest_dist <= plausible_contact_dist:
            agent_collision_flags[partner_key][closest_frame] = True
            native_fix_applied = True

    instance = RawTrajectoryInstance(
        dataset=DATASET_NAME,
        instance_id=f"{clip_name}__{role}__r{record_idx:04d}",
        times=ego_times, ego_box=ego_box, agent_boxes=agent_boxes,
        agent_collision_flags=agent_collision_flags,
        lane_graph_available=lane_graph_available,
        ego_lane_change_times=ego_lane_change_times if lane_graph_available else None,
        agent_lane_change_times=agent_lane_change_times if lane_graph_available else None,
        agent_classes=agent_classes,
    )
    # extra, non-schema attributes (same pattern as dataset.py's
    # batch.instance_ids/batch.collision_label) -- not part of
    # RawTrajectoryInstance's dataclass fields, so every other adapter and
    # the dataset-agnostic core pipeline stay untouched.
    instance.native_collision_label = int(native_is_collision)
    instance.native_collision_intensity = collision_intensity
    instance.native_collision_fix_applied = int(native_fix_applied)
    return instance


def build_instances_from_file(
    tfrecord_path: str, cfg: FutureRiskConfig = DEFAULT_CONFIG, roles: Tuple[str, ...] = ROLES,
    record_stride: int = 1,
) -> List[RawTrajectoryInstance]:
    """record_stride > 1 keeps only every Nth record (0, N, 2N, ...) --
    consecutive records within one file typically share `scenario_id`
    (overlapping time-offset snapshots of the same underlying incident, per
    the module docstring), so striding trades a bounded loss of redundant
    near-duplicate windows for a proportional cut in total instance count.
    Default 1 keeps every record (no subsampling)."""
    clip_name = Path(tfrecord_path).stem
    instances: List[RawTrajectoryInstance] = []
    for record_idx, feat in enumerate(load_records(tfrecord_path)):
        if record_idx % record_stride != 0:
            continue
        for role in roles:
            inst = _build_role_instance(feat, role, clip_name, record_idx, cfg)
            if inst is not None:
                instances.append(inst)
    return instances


def find_motion_files(root_dir: str) -> List[str]:
    return sorted(glob.glob(str(Path(root_dir) / "**" / "*.tfrecords"), recursive=True))


def build_instances_from_directory(
    root_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG, roles: Tuple[str, ...] = ROLES,
    record_stride: int = 1,
) -> List[RawTrajectoryInstance]:
    instances: List[RawTrajectoryInstance] = []
    for path in find_motion_files(root_dir):
        instances.extend(build_instances_from_file(path, cfg=cfg, roles=roles, record_stride=record_stride))
    return instances


def preprocess_deepaccident(
    motion_data_dir: str, output_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG,
    dataset_name: str = DATASET_NAME, roles: Tuple[str, ...] = ROLES, record_stride: int = 1,
):
    """End-to-end: build RawTrajectoryInstances from every .tfrecords file
    under `motion_data_dir`, then run the existing (dataset-agnostic)
    `data_pipeline.preprocess_dataset`."""
    from ..data_pipeline import preprocess_dataset

    instances = build_instances_from_directory(motion_data_dir, cfg=cfg, roles=roles, record_stride=record_stride)
    if not instances:
        raise ValueError(f"No usable trajectory instances built from {motion_data_dir}")
    return preprocess_dataset(dataset_name, instances, output_dir, cfg=cfg)


def collect_native_collision_labels(instances) -> Dict[str, dict]:
    """Pull the `native_collision_label`/`native_collision_intensity`
    attributes `_build_role_instance` attaches to each instance into a
    plain, JSON-serializable dict keyed by instance_id -- meant to be saved
    once alongside the preprocessed cache (see
    scripts/preprocess_deepaccident.py) so evaluation never has to re-parse
    the source .tfrecords just to get an independent ground-truth label."""
    out = {}
    for inst in instances:
        out[inst.instance_id] = {
            "collision_label": getattr(inst, "native_collision_label", None),
            "collision_intensity": getattr(inst, "native_collision_intensity", None),
            "collision_fix_applied": getattr(inst, "native_collision_fix_applied", None),
        }
    return out
