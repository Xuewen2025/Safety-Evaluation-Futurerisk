"""Bench2Drive -> RawTrajectoryInstance adapter.

Turns the raw `rethinklab/Bench2Drive` clip archives (one `.tar.gz` per
scenario clip, e.g. from the "Mini" 10-clip subset) into
`data_pipeline.RawTrajectoryInstance` objects that plug directly into the
existing preprocessing pipeline (`data_pipeline.preprocess_dataset`).

Schema notes verified against an actual download (10 Bench2Drive-Mini
clips, `rethinklab/Bench2Drive` on Hugging Face) rather than assumed from
documentation, since the dataset's own docs don't specify several details
this adapter depends on:

- Each clip extracts to `<ClipName>/{anno,camera,lidar,radar,
  expert_assessment}/`. Only `anno/*.json.gz` is used here -- one gzipped
  JSON per simulation frame, holding the ego state plus a `bounding_boxes`
  list (one dict per nearby actor: ego included, as `class="ego_vehicle"`).
- **Frame rate is 10Hz** (dt=0.1s) -- not documented anywhere, verified
  empirically here by dividing consecutive-frame ego displacement (from
  `bounding_boxes[ego]['center']`, not the top-level x/y -- see next point)
  by ego speed and confirming it lands on 0.0998-0.1004s consistently.
- The frame's **top-level `x`/`y` are noisy** (consistent with simulated
  GPS/localization noise added for realism) -- consecutive-frame distances
  computed from them don't match `speed` at all. `bounding_boxes[*]
  ['center']` is the exact simulator ground-truth box center and is what's
  used for every agent including ego.
- `extent` is CARLA's convention of **half-extents** along the box's own
  (forward, lateral, vertical) axes -- confirmed by checking it against the
  real-world dimensions of the annotated vehicle model (e.g. `extent[0]
  =2.446` for a `vehicle.lincoln.mkz_2020`, whose real full length is
  ~4.9m). `BoxTrajectory` wants full lengths, so this adapter doubles them.
- The per-box `rotation` field (and the frame's top-level `theta` for ego)
  could **not** be reconciled with a consistent, verifiable heading
  convention (see the long comment in `geometry.heading_from_positions`)
  -- heading is instead derived from consecutive box centers for every
  agent, ego included, via that function.
- There is **no explicit collision flag anywhere** in the dataset (checked
  against the dataset's own `docs/anno.md`) -- collisions are instead
  detected geometrically, reusing `geometry.boxes_overlap` on ego vs. each
  agent's box at matching frames (delta=0), exactly like the rest of this
  package's own collision-overlap logic.
- `road_id`/`lane_id` per box are used as a **lightweight lane-change
  proxy**: a change in that tuple over consecutive frames is treated as
  "changing/merging lanes at this time" for S3.3's merge-window condition
  (Algorithm 1's actual step needs full lane-graph topology, which isn't in
  this per-clip download -- see `Bench2Drive-Map`, not fetched here; this
  proxy is a reasonable stand-in but can miss/misfire at complex junctions).
- **Actor ids get recycled within one recorded episode** -- confirmed
  directly: a pedestrian id was seen parked off-map (z ~ -100m, a common
  CARLA "inactive actor" placement) for ~2s, then instantly "teleported"
  ~67m to a real spawn position at the very next frame under the same id,
  because an unrelated pedestrian later reused it. `_segment_agent_positions`
  splits a track wherever consecutive frames imply an unphysical speed, so
  this doesn't silently splice two unrelated actors into one track.
- `expert_assessment/*.npz` (one flat, undocumented ~1500-float vector per
  frame, presumably internal features of the Think2Drive RL expert used to
  collect the data) is intentionally **not used** -- there is no way to
  verify what its fields mean without the collection source code, and nothing
  else in this dataset requires it.

Windowing: S3.1's whole method assumes one trajectory instance spans a
fixed horizon H with one shared ego trajectory and every surrounding agent
observed over that *same* window -- a raw ~20s clip is not itself one
"instance" in that sense. This adapter slides a fixed H-second window
(`cfg.horizon_s`, default 5s = 51 frames at 10Hz) over each clip; an agent
is included in a given window only if it is tracked (contiguously) for the
window's *entire* length, so every RawTrajectoryInstance this produces has
one shared time axis for the ego box and every included agent box, as
`data_pipeline.preprocess_instance` requires.
"""
from __future__ import annotations

import gzip
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import FutureRiskConfig, DEFAULT_CONFIG
from ..data_pipeline import RawTrajectoryInstance
from ..geometry import BoxTrajectory, boxes_overlap, heading_from_positions

DATASET_NAME = "Bench2Drive"

# A track shorter than this (at cfg.dt=0.1s, 10 frames = 1s) carries too
# little signal to build a meaningful interaction trace and is dropped.
MIN_TRACK_FRAMES = 10

# Consecutive-frame displacement beyond this implied speed is treated as an
# id-recycling teleport rather than real motion (see
# _segment_agent_positions) -- generous enough to never clip genuine
# vehicle motion (144 km/h) while easily catching splices that are orders
# of magnitude faster.
MAX_AGENT_SPEED_MPS = 40.0

_STATIC_CLASSES = {"traffic_light", "traffic_sign"}   # not dynamic interactive agents
_BASE_TYPE_CLASS_MAP = {"bicycle": "cyclist", "motorcycle": "cyclist"}


def _agent_class_for(bbox: dict) -> str:
    """Bench2Drive `class`/`base_type` -> our agent-class vocabulary
    (config.PhysicalPriorParams keys: vehicle/truck/bus/cyclist/pedestrian)."""
    if bbox["class"] == "walker":
        return "pedestrian"
    base_type = (bbox.get("base_type") or "").strip().lower()
    return _BASE_TYPE_CLASS_MAP.get(base_type, "vehicle")


def _longest_contiguous_run(values: List[int]) -> Tuple[int, int]:
    """Return the (start, end) slice (into `values`, half-open) of the
    longest run where consecutive entries differ by exactly 1. Used both
    for a clip's own frame-index contiguity and for a single agent's
    presence-position contiguity -- same shape of problem either way."""
    if not values:
        return 0, 0
    best = (0, 1)
    run_start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1] + 1:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = i
    return best


def load_clip_frames(tar_path: str) -> Tuple[str, Dict[int, dict]]:
    """Single sequential pass over one clip .tar.gz, reading only
    `anno/*.json.gz` members (camera/lidar/radar/expert_assessment bytes
    are skipped, not decompressed). Returns (clip_name, {frame_index:
    parsed_json_dict})."""
    frames: Dict[int, dict] = {}
    clip_name: Optional[str] = None
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
    if not frames:
        raise ValueError(f"No anno/*.json.gz frames found in {tar_path}")
    return clip_name, frames


def _extract_ego_track(
    frames: Dict[int, dict], idxs: List[int], cfg: FutureRiskConfig,
) -> Tuple[BoxTrajectory, List[Tuple]]:
    xs, ys, zs, lengths, widths, heights, road_lane = [], [], [], [], [], [], []
    for idx in idxs:
        bboxes = frames[idx]["bounding_boxes"]
        ego = next((b for b in bboxes if b["class"] == "ego_vehicle"), None)
        if ego is None:
            raise ValueError(f"frame {idx} is missing an ego_vehicle bounding box")
        cx, cy, cz = ego["center"]
        el, ew, eh = ego["extent"]
        xs.append(cx); ys.append(cy); zs.append(cz)
        lengths.append(2 * el); widths.append(2 * ew); heights.append(2 * eh)
        road_lane.append((ego.get("road_id"), ego.get("lane_id")))
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    heading = heading_from_positions(xs, ys, dt=cfg.dt)
    box = BoxTrajectory(xs, ys, zs, heading, np.array(lengths), np.array(widths), np.array(heights))
    return box, road_lane


def _collect_agent_bboxes_by_position(
    frames: Dict[int, dict], idxs: List[int],
) -> Dict[str, Dict[int, dict]]:
    """agent_id -> {position (0-based index into `idxs`): bbox dict}, for
    every non-ego, non-static-infrastructure actor seen anywhere in `idxs`."""
    out: Dict[str, Dict[int, dict]] = {}
    for pos, idx in enumerate(idxs):
        for b in frames[idx]["bounding_boxes"]:
            if b["class"] == "ego_vehicle" or b["class"] in _STATIC_CLASSES:
                continue
            out.setdefault(str(b["id"]), {})[pos] = b
    return out


@dataclass
class _AgentTrack:
    agent_class: str
    run_start: int              # position index into the clip's `idxs`, inclusive
    run_end: int                 # position index into the clip's `idxs`, exclusive
    box: BoxTrajectory            # length == run_end - run_start
    road_lane: List[Tuple]        # same length


def _segment_agent_positions(
    bbox_by_pos: Dict[int, dict], dt: float, max_speed_mps: float = MAX_AGENT_SPEED_MPS,
) -> List[List[int]]:
    """Split one actor id's observed positions into runs of physically
    plausible continuous motion.

    A run breaks wherever consecutive positions are non-adjacent (a missing
    frame) OR the center-to-center displacement between them implies a
    speed beyond `max_speed_mps`. The latter guards against a real gotcha
    confirmed against the actual Bench2Drive-Mini download: CARLA recycles
    actor ids within one recorded episode -- e.g. a pedestrian id was found
    parked off-map (z~-100m, a common "inactive actor" placement) for its
    first ~2s, then instantly "teleported" ~67m to a real spawn position at
    the very next frame with the same id, because an unrelated pedestrian
    reused that id. Without this check, `_build_agent_track` would splice
    two unrelated actors into one bogus track and inject a spurious
    ~700m/s velocity spike right at the splice point, corrupting heading,
    evasive severity, and interaction-type classification there.

    Returns a list of contiguous position-index runs (each ascending).
    """
    positions = sorted(bbox_by_pos.keys())
    if not positions:
        return []
    runs = [[positions[0]]]
    max_disp = max_speed_mps * dt
    for prev_pos, pos in zip(positions, positions[1:]):
        if pos == prev_pos + 1:
            prev_c = bbox_by_pos[prev_pos]["center"]
            cur_c = bbox_by_pos[pos]["center"]
            disp = sum((a - b) ** 2 for a, b in zip(cur_c, prev_c)) ** 0.5
            if disp <= max_disp:
                runs[-1].append(pos)
                continue
        runs.append([pos])
    return runs


def _build_agent_track(
    bbox_by_pos: Dict[int, dict], cfg: FutureRiskConfig, min_track_frames: int,
) -> Optional[_AgentTrack]:
    runs = _segment_agent_positions(bbox_by_pos, dt=cfg.dt)
    if not runs:
        return None
    run_positions = max(runs, key=len)
    if len(run_positions) < min_track_frames:
        return None

    xs, ys, zs, lengths, widths, heights, road_lane = [], [], [], [], [], [], []
    agent_class = None
    for pos in run_positions:
        b = bbox_by_pos[pos]
        if agent_class is None:
            agent_class = _agent_class_for(b)
        cx, cy, cz = b["center"]
        el, ew, eh = b["extent"]
        xs.append(cx); ys.append(cy); zs.append(cz)
        lengths.append(2 * el); widths.append(2 * ew); heights.append(2 * eh)
        road_lane.append((b.get("road_id"), b.get("lane_id")))

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    heading = heading_from_positions(xs, ys, dt=cfg.dt)
    box = BoxTrajectory(xs, ys, zs, heading, np.array(lengths), np.array(widths), np.array(heights))
    # run_positions is contiguous by construction (see _segment_agent_positions),
    # so its first/last+1 entries are exactly the (run_start, run_end) bounds.
    return _AgentTrack(
        agent_class=agent_class, run_start=run_positions[0], run_end=run_positions[-1] + 1,
        box=box, road_lane=road_lane,
    )


def _slice_box(box: BoxTrajectory, start: int, end: int) -> BoxTrajectory:
    return BoxTrajectory(
        box.x[start:end], box.y[start:end], box.z[start:end], box.heading[start:end],
        box.length[start:end], box.width[start:end], box.height[start:end],
    )


def _lane_change_event_times(road_lane_seq: List[Tuple], times: np.ndarray) -> np.ndarray:
    """Timestamps where the (road_id, lane_id) tuple changes between
    consecutive frames -- the lane-change/merge proxy described in this
    module's docstring."""
    out = []
    prev = None
    for rl, t in zip(road_lane_seq, times):
        if prev is not None and rl != prev and None not in rl and None not in prev:
            out.append(t)
        prev = rl
    return np.array(out, dtype=float)


def build_instances_from_clip(
    tar_path: str,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    window_stride_frames: Optional[int] = None,
    min_track_frames: int = MIN_TRACK_FRAMES,
    include_empty_windows: bool = False,
) -> List[RawTrajectoryInstance]:
    """Build every fixed-H-second-window RawTrajectoryInstance out of one
    Bench2Drive clip archive. See module docstring for the windowing
    rationale.

    window_stride_frames: stride between window starts, in frames;
    defaults to a non-overlapping window (== the window length itself).
    Pass e.g. window_frames // 2 for 50%-overlapping windows (more
    instances per clip, at the cost of adjacent instances sharing data).
    include_empty_windows: if True, keep windows where no agent is
    contiguously tracked for the whole window (R=0 by construction via
    eq:agg's empty product) instead of skipping them -- off by default so
    a small clip set isn't dominated by trivial no-interaction windows.
    """
    clip_name, frames = load_clip_frames(tar_path)
    all_idxs = sorted(frames.keys())
    start, end = _longest_contiguous_run(all_idxs)
    idxs = all_idxs[start:end]
    if len(idxs) < 2:
        return []

    ego_box, ego_road_lane = _extract_ego_track(frames, idxs, cfg)

    bboxes_by_agent = _collect_agent_bboxes_by_position(frames, idxs)
    tracks: Dict[str, _AgentTrack] = {}
    for agent_id, bbox_by_pos in bboxes_by_agent.items():
        track = _build_agent_track(bbox_by_pos, cfg, min_track_frames=min_track_frames)
        if track is not None:
            tracks[agent_id] = track

    window_frames = int(round(cfg.horizon_s / cfg.dt)) + 1
    stride = window_stride_frames or window_frames
    n = len(idxs)

    instances: List[RawTrajectoryInstance] = []
    for w_start in range(0, max(n - window_frames + 1, 0), stride):
        w_end = w_start + window_frames
        times = np.arange(window_frames) * cfg.dt
        ego_slice = _slice_box(ego_box, w_start, w_end)
        idx_id = np.arange(window_frames)

        agent_boxes, collision_flags, lane_change_times, agent_classes = {}, {}, {}, {}
        for agent_id, track in tracks.items():
            if track.run_start <= w_start and track.run_end >= w_end:
                local_start, local_end = w_start - track.run_start, w_end - track.run_start
                agent_slice = _slice_box(track.box, local_start, local_end)
                agent_boxes[agent_id] = agent_slice
                collision_flags[agent_id] = boxes_overlap(
                    ego_slice, idx_id, agent_slice, idx_id, check_vertical=True,
                )
                lane_change_times[agent_id] = _lane_change_event_times(
                    track.road_lane[local_start:local_end], times,
                )
                agent_classes[agent_id] = track.agent_class

        if not agent_boxes and not include_empty_windows:
            continue

        instances.append(RawTrajectoryInstance(
            dataset=DATASET_NAME,
            instance_id=f"{clip_name}__w{idxs[w_start]:05d}",
            times=times,
            ego_box=ego_slice,
            agent_boxes=agent_boxes,
            agent_collision_flags=collision_flags,
            lane_graph_available=True,
            ego_lane_change_times=_lane_change_event_times(ego_road_lane[w_start:w_end], times),
            agent_lane_change_times=lane_change_times,
            agent_classes=agent_classes,
        ))

    return instances


def find_clip_archives(root_dir: str) -> List[Path]:
    return sorted(Path(root_dir).glob("*.tar.gz"))


def build_instances_from_directory(
    root_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG, **kwargs,
) -> List[RawTrajectoryInstance]:
    """Run `build_instances_from_clip` over every clip archive in a
    directory (e.g. the Bench2Drive-Mini download directory) and
    concatenate the results."""
    instances: List[RawTrajectoryInstance] = []
    for tar_path in find_clip_archives(root_dir):
        instances.extend(build_instances_from_clip(str(tar_path), cfg=cfg, **kwargs))
    return instances


def preprocess_bench2drive(
    archive_dir: str, output_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG,
    dataset_name: str = DATASET_NAME, **kwargs,
):
    """End-to-end: read every clip archive under `archive_dir`, build
    RawTrajectoryInstances, then run the existing
    `data_pipeline.preprocess_dataset` (threshold fitting + T_evt/tau_phy
    generation + caching) exactly as it would for any other dataset."""
    from ..data_pipeline import preprocess_dataset

    instances = build_instances_from_directory(archive_dir, cfg=cfg, **kwargs)
    if not instances:
        raise ValueError(f"No usable trajectory instances built from {archive_dir}")
    return preprocess_dataset(dataset_name, instances, output_dir, cfg=cfg)
