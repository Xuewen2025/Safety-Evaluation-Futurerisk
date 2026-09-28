"""TRACE (Salsabil & Elbaum, arXiv:2604.22068, FSE'26; github.com/
NahianSalsabil/carla-benchmark, MIT license) -> RawTrajectoryInstance
adapter.

TRACE reconstructs real NHTSA crash reports into CARLA scenario
*specifications*: 52 two-vehicle scenarios under `Benchmark/`, each a
`simulations/<crash_id>.json` with, per vehicle, a `start_position` (x,y,z),
a single scalar `speed_kmh`, an initial `yaw` (degrees), and a sparse
ordered list of `trajectory` waypoints ([x,y], no timestamps) -- plus one
shared `crash_point` (x,y) for the whole scenario. `Benchmark/summary/*.txt`
holds the parsed NHTSA report text (vehicle `Model`, speeds in MPH, impact
geometry, etc.).

Why this adapter does kinematic reconstruction instead of replaying CARLA
-------------------------------------------------------------------------
Turning a `simulations/*.json` scenario into actual per-tick physics
requires literally running it inside CARLA 0.9.15 (the repo's own
`Benchmark/README.md`: "you need to use the docker image ... run
`./run_benchmark.sh <crash_id>`", requiring Docker + an NVIDIA Container
Toolkit + an X11 server). That is real infrastructure work with its own
fidelity payoff (an independent, non-tautological collision detector via
CARLA's own collision sensor), deliberately deferred as a follow-up -- see
`experiment_design.txt`. This adapter instead reconstructs an approximate
dense trajectory directly from the JSON's sparse fields, so the existing
dataset-agnostic pipeline (`data_pipeline.preprocess_dataset`,
`geometry.boxes_overlap`, every SSM/CARM baseline) can run today. Every
approximation below is a disclosed judgment call, not a hidden one, and
should be flagged as a limitation in any reported numbers.

Reconstruction judgment calls
------------------------------
1. **Constant speed along an extended polyline.** Each vehicle's path is
   built as `[start_position] + trajectory + [crash_point]` (consecutive
   near-duplicate points, < 0.3m apart, collapsed). The raw `trajectory`
   list frequently ends far short of `crash_point` (checked directly:
   distance from the last waypoint to `crash_point` ranges ~4-34m across
   sampled scenarios) -- appending `crash_point` is not an invented point,
   it is the scenario's own annotated collision location, and it is the
   only way the two vehicles' reconstructed paths actually reach the same
   coordinate. The vehicle then moves along this polyline's arc length at
   the single constant `speed_kmh` given -- there is no per-tick speed
   profile (e.g. braking) in the source data, so evasive-severity / a_max
   signal derived downstream from this reconstruction is dominated by path
   curvature, not braking, and is weaker evidence than DeepAccident's own
   simulated physics. Document this whenever `evasive_severity`/a_max
   numbers from this adapter are reported.
2. **Inter-vehicle timing solved for simultaneous impact.** The JSON gives
   no relative start-time offset between the two vehicles. Each vehicle's
   own point-to-point traversal time (`T_i = arc_length_i / speed_i`) was
   checked directly across sampled scenarios and lands within about a
   second of the other vehicle's (e.g. 7.79s vs 6.98s) -- consistent with
   the reconstruction pipeline having picked (position, speed) pairs that
   already roughly line up in time, but not exactly. We solve for the
   minimal additional assumption needed to reproduce the annotated
   ground-truth collision: let `T_impact = max(T_1, T_2)`; the vehicle with
   the shorter travel time waits at its `start_position` (heading held at
   the low-speed fallback) until `T_impact - T_i`, then moves at constant
   speed, so both vehicles reach `crash_point` at exactly the same tick.
   Without this, naively starting both vehicles at t=0 can produce a near
   miss (paths cross in space but not in time) for a scenario whose ground
   truth is an actual collision.
3. **Heading from position, not the given `yaw`.** Same convention as
   `adapters/bench2drive.py` (see `geometry.heading_from_positions`'s
   docstring): heading is derived from consecutive reconstructed positions
   via atan2, with the given initial `yaw` only used implicitly through
   `heading_from_positions`'s nearest-valid-neighbor fill during the
   pre-motion "waiting" segment (judgment call 2), which needs some
   fallback heading. This avoids trusting a raw orientation field whose
   axis convention was not independently verified against motion, exactly
   the reason that project-wide convention exists.
4. **Box extents are generic per-class defaults, not per-vehicle.** The
   JSON carries no bounding-box size. `Benchmark/summary/<id>.txt`'s
   `Model:` line (e.g. "Honda Odyssey", "Honda 750 cc or greater") is
   parsed into this project's class vocabulary
   (vehicle/truck/bus/cyclist/other) via `_classify_vehicle_model`, and each
   class gets one fixed generic (length, width, height) in `_CLASS_BOX_M`
   -- not the real model's dimensions, which are not available here.
5. **Collision label is geometric (SAT overlap), same convention as
   Bench2Drive, not an independent native label like DeepAccident's.**
   `crash_point` anchors *where* the paths are extended to meet (judgment
   call 1) and `boxes_overlap` (this project's existing SAT+vertical test)
   determines *whether/when* the two reconstructed boxes actually intersect
   at a shared tick, exactly like `adapters/bench2drive.py`. Because the
   reconstruction explicitly routes both paths through `crash_point`, the
   overlap check is close to tautological with the ground truth by
   construction for the window that actually contains the impact -- unlike
   DeepAccident's own simulator-native label, this is not an independent
   signal, and this caveat should be reported alongside any TRACE numbers.

Windowing
---------
Each scenario yields exactly one two-agent trajectory (vehicle1 as ego,
vehicle2 as the sole surrounding agent -- an arbitrary but fixed and
reproducible role assignment), of length `T_impact + a short post-impact
tail` at `cfg.dt`. As in `adapters/bench2drive.py`, a fixed `cfg.horizon_s`
window is slid over this single trajectory (default stride = a quarter of
the window, denser than Bench2Drive's non-overlapping default, since only
52 underlying scenarios are available and multiple overlapping windows per
scenario are needed for a usable sample size) to produce multiple
`RawTrajectoryInstance` objects per scenario -- windows containing the
impact tick are positive, windows entirely before it are negative, exactly
the same mechanism (no separate negative-mining logic needed). All windows
from one scenario are correlated (52 independent underlying events, not one
per window) -- downstream analysis must report and account for this (e.g.
cluster the bootstrap/significance test by `crash_id`, not by window).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import FutureRiskConfig, DEFAULT_CONFIG
from ..data_pipeline import RawTrajectoryInstance
from ..geometry import BoxTrajectory, boxes_overlap, heading_from_positions

DATASET_NAME = "TRACE"

# Consecutive polyline points closer than this are collapsed to one (avoids
# zero-length arc segments, e.g. when `start_position` already coincides
# with `trajectory[0]`, or `trajectory[-1]` already coincides with
# `crash_point`).
DEDUP_DIST_M = 0.3

# generic full-extent (length, width, height) box sizes in meters, per this
# project's agent-class vocabulary -- see judgment call 4.
_CLASS_BOX_M: Dict[str, Tuple[float, float, float]] = {
    "vehicle": (4.7, 1.85, 1.5),
    "truck": (6.5, 2.3, 2.5),
    "bus": (10.0, 2.5, 3.2),
    "cyclist": (2.2, 0.9, 1.3),
    "other": (4.7, 1.85, 1.5),
}

_TRUCK_KEYWORDS = ("truck", "pickup", "f-150", "f150", "silverado", "ram ", "sierra", "tundra", "semi", "tractor")
_BUS_KEYWORDS = ("bus", "coach", "shuttle")
_MOTORCYCLE_KEYWORDS = ("cc or greater", " cc", "motorcycle", "moped", "scooter", "harley")


def _classify_vehicle_model(model_text: str) -> str:
    """`Benchmark/summary/<id>.txt` `Model:` free text -> this project's
    agent-class vocabulary. Falls back to "vehicle" (the modal case in a
    highway-crash sample) rather than "other" for anything not matched, so
    an ordinary car model this heuristic doesn't recognize still gets a
    car-shaped box rather than the (also car-shaped, coincidentally
    identical here) most-conservative bucket."""
    text = (model_text or "").strip().lower()
    if any(k in text for k in _MOTORCYCLE_KEYWORDS):
        return "cyclist"
    if any(k in text for k in _BUS_KEYWORDS):
        return "bus"
    if any(k in text for k in _TRUCK_KEYWORDS):
        return "truck"
    return "vehicle"


_MODEL_LINE_RE = re.compile(r"Vehicle (\d+):\s*\n\s*Model:\s*(.+)")


def parse_vehicle_classes(summary_path: Path) -> Dict[int, str]:
    """Parse `Benchmark/summary/<id>.txt` for each `Vehicle N:` block's
    `Model:` line, returning {vehicle_number: agent_class}. Missing/
    unparseable entries default to "vehicle" (see `_classify_vehicle_model`)."""
    if not summary_path.exists():
        return {}
    text = summary_path.read_text(errors="ignore")
    out: Dict[int, str] = {}
    for match in _MODEL_LINE_RE.finditer(text):
        vnum = int(match.group(1))
        out[vnum] = _classify_vehicle_model(match.group(2))
    return out


def _dedup_polyline(points: List[Tuple[float, float]], min_dist: float = DEDUP_DIST_M) -> List[Tuple[float, float]]:
    out = [points[0]]
    for p in points[1:]:
        if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_dist:
            out.append(p)
    return out


def _arc_length_table(points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cumulative arc length per vertex, xs, ys)."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, xs, ys


def _sample_at_arclength(cum: np.ndarray, xs: np.ndarray, ys: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (x, y) at arc-length position(s) `s` along
    the polyline described by `cum`/`xs`/`ys`. `s` is clamped to
    [0, cum[-1]] (holds at the last vertex past the path's end)."""
    s = np.clip(s, 0.0, cum[-1])
    x = np.interp(s, cum, xs)
    y = np.interp(s, cum, ys)
    return x, y


# A vehicle with speed_kmh below this is treated as stationary (judgment
# call 1a) -- confirmed against a real scenario (510121) where a
# speed_kmh=0 vehicle nonetheless carries a non-degenerate `trajectory`
# list (presumably a reconstruction-pipeline artifact for an at-rest
# vehicle, e.g. "was stopped and got rear-ended"): a 0-speed vehicle cannot
# have actually traversed that polyline under its own power before impact,
# so it is held at `start_position` for the whole scenario instead.
STATIONARY_SPEED_MPS = 0.1


@dataclass
class _VehiclePath:
    cum: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    speed_mps: float
    travel_time_s: float          # time to traverse the full (crash-point-extended) polyline
    start_xy: Tuple[float, float]
    yaw_rad: float                # given initial yaw, only used if stationary
    is_stationary: bool = False


def _build_vehicle_path(vehicle: dict) -> _VehiclePath:
    sp = vehicle["start_position"]
    speed_mps = float(vehicle["speed_kmh"]) / 3.6
    yaw_rad = float(np.radians(vehicle.get("yaw", 0.0)))
    if speed_mps < STATIONARY_SPEED_MPS:
        xs, ys = np.array([sp["x"]]), np.array([sp["y"]])
        return _VehiclePath(
            cum=np.array([0.0]), xs=xs, ys=ys, speed_mps=0.0, travel_time_s=0.0,
            start_xy=(sp["x"], sp["y"]), yaw_rad=yaw_rad, is_stationary=True,
        )
    points = [(sp["x"], sp["y"])] + [tuple(p) for p in vehicle["trajectory"]]
    points = _dedup_polyline(points)
    cum, xs, ys = _arc_length_table(points)
    travel_time = float(cum[-1] / speed_mps)
    return _VehiclePath(cum=cum, xs=xs, ys=ys, speed_mps=speed_mps, travel_time_s=travel_time,
                         start_xy=(sp["x"], sp["y"]), yaw_rad=yaw_rad)


def _extend_path_to_point(path: _VehiclePath, point_xy: Tuple[float, float]) -> _VehiclePath:
    """Append `point_xy` (the scenario's `crash_point`) as the final vertex
    if it isn't already effectively the endpoint (judgment call 1). A
    stationary vehicle (judgment call 1a) is left untouched -- it never
    moves, so extending its polyline is meaningless."""
    if path.is_stationary:
        return path
    last = (path.xs[-1], path.ys[-1])
    if np.hypot(point_xy[0] - last[0], point_xy[1] - last[1]) < DEDUP_DIST_M:
        return path
    xs = np.append(path.xs, point_xy[0])
    ys = np.append(path.ys, point_xy[1])
    extra = np.hypot(point_xy[0] - last[0], point_xy[1] - last[1])
    cum = np.append(path.cum, path.cum[-1] + extra)
    travel_time = float(cum[-1] / path.speed_mps)
    return _VehiclePath(cum=cum, xs=xs, ys=ys, speed_mps=path.speed_mps, travel_time_s=travel_time,
                         start_xy=path.start_xy, yaw_rad=path.yaw_rad)


def reconstruct_scenario_boxes(
    scenario: dict,
    vehicle_classes: Dict[int, str],
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    post_impact_tail_s: float = 1.0,
) -> Tuple[BoxTrajectory, BoxTrajectory, np.ndarray, str, str]:
    """Kinematically reconstruct both vehicles onto one shared time axis
    (judgment calls 1-4). Returns (box_vehicle1, box_vehicle2, times,
    class_vehicle1, class_vehicle2). `times[0] == 0.0`."""
    crash_xy = (scenario["crash_point"]["x"], scenario["crash_point"]["y"])

    path1 = _extend_path_to_point(_build_vehicle_path(scenario["vehicle1"]), crash_xy)
    path2 = _extend_path_to_point(_build_vehicle_path(scenario["vehicle2"]), crash_xy)

    t_impact = max(path1.travel_time_s, path2.travel_time_s)
    # judgment call 2: the shorter-travel-time vehicle waits at start_xy
    # until t_impact - its own travel time, so both arrive at crash_xy
    # simultaneously.
    wait1 = t_impact - path1.travel_time_s
    wait2 = t_impact - path2.travel_time_s

    dt = cfg.dt
    duration = t_impact + post_impact_tail_s
    times = np.arange(0.0, duration + 1e-9, dt)

    def _box_for(path: _VehiclePath, wait: float, agent_class: str, z: float) -> BoxTrajectory:
        if path.is_stationary:
            x = np.full(len(times), path.start_xy[0])
            y = np.full(len(times), path.start_xy[1])
            heading = np.full(len(times), path.yaw_rad)
        else:
            elapsed = np.clip(times - wait, 0.0, None)
            s = elapsed * path.speed_mps
            x, y = _sample_at_arclength(path.cum, path.xs, path.ys, s)
            heading = heading_from_positions(x, y, dt=dt, min_speed=0.3, fallback_heading=np.full(len(times), path.yaw_rad))
        length, width, height = _CLASS_BOX_M.get(agent_class, _CLASS_BOX_M["vehicle"])
        return BoxTrajectory(
            x=x, y=y, z=np.full_like(x, z), heading=heading,
            length=length, width=width, height=height,
        )

    cls1 = vehicle_classes.get(1, "vehicle")
    cls2 = vehicle_classes.get(2, "vehicle")
    z1 = float(scenario["vehicle1"]["start_position"].get("z", 0.5))
    z2 = float(scenario["vehicle2"]["start_position"].get("z", 0.5))
    box1 = _box_for(path1, wait1, cls1, z1)
    box2 = _box_for(path2, wait2, cls2, z2)
    return box1, box2, times, cls1, cls2


def build_instances_from_scenario(
    simulation_path: str,
    summary_path: Optional[str] = None,
    cfg: FutureRiskConfig = DEFAULT_CONFIG,
    window_stride_frames: Optional[int] = None,
    post_impact_tail_s: float = 1.0,
) -> List[RawTrajectoryInstance]:
    """Build every fixed-H-second-window RawTrajectoryInstance out of one
    TRACE `simulations/<crash_id>.json` scenario (see module docstring,
    "Windowing")."""
    crash_id = Path(simulation_path).stem.replace("simulation_", "")
    scenario = json.loads(Path(simulation_path).read_text())

    summary = Path(summary_path) if summary_path is not None else Path(simulation_path).resolve().parents[1] / "summary" / f"summary_{crash_id}.txt"
    vehicle_classes = parse_vehicle_classes(summary)

    ego_box, agent_box, times, ego_cls, agent_cls = reconstruct_scenario_boxes(
        scenario, vehicle_classes, cfg=cfg, post_impact_tail_s=post_impact_tail_s,
    )

    idx = np.arange(len(times))
    collision_flags_full = boxes_overlap(ego_box, idx, agent_box, idx, check_vertical=True)

    window_frames = int(round(cfg.horizon_s / cfg.dt)) + 1
    stride = window_stride_frames or max(window_frames // 4, 1)
    n = len(times)

    instances: List[RawTrajectoryInstance] = []
    starts = list(range(0, max(n - window_frames + 1, 0), stride))
    if not starts:
        # scenario shorter than one full window: use the single, shorter,
        # available span rather than dropping the scenario entirely.
        starts = [0]
        window_frames = n

    for w_start in starts:
        w_end = min(w_start + window_frames, n)
        if w_end - w_start < 2:
            continue
        window_times = times[w_start:w_end] - times[w_start]

        def _slice(box: BoxTrajectory) -> BoxTrajectory:
            sl = slice(w_start, w_end)
            return BoxTrajectory(box.x[sl], box.y[sl], box.z[sl], box.heading[sl], box.length[sl], box.width[sl], box.height[sl])

        ego_slice = _slice(ego_box)
        agent_slice = _slice(agent_box)
        flags = collision_flags_full[w_start:w_end]

        instances.append(RawTrajectoryInstance(
            dataset=DATASET_NAME,
            instance_id=f"trace_{crash_id}__w{w_start:04d}",
            times=window_times,
            ego_box=ego_slice,
            agent_boxes={"vehicle2": agent_slice},
            agent_collision_flags={"vehicle2": flags},
            lane_graph_available=False,
            agent_classes={"vehicle2": agent_cls},
        ))

    return instances


def find_scenario_files(root_dir: str) -> List[Path]:
    return sorted(Path(root_dir).glob("simulations/simulation_*.json"))


def build_instances_from_directory(
    root_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG, **kwargs,
) -> List[RawTrajectoryInstance]:
    """Run `build_instances_from_scenario` over every `simulations/*.json`
    file under a TRACE `Benchmark/` directory and concatenate the results."""
    instances: List[RawTrajectoryInstance] = []
    for sim_path in find_scenario_files(root_dir):
        instances.extend(build_instances_from_scenario(str(sim_path), cfg=cfg, **kwargs))
    return instances


def preprocess_trace(
    benchmark_dir: str, output_dir: str, cfg: FutureRiskConfig = DEFAULT_CONFIG,
    dataset_name: str = DATASET_NAME, **kwargs,
):
    """End-to-end: read every scenario under `benchmark_dir` (a TRACE
    `Benchmark/` directory), build RawTrajectoryInstances, then run the
    existing `data_pipeline.preprocess_dataset` exactly as for any other
    dataset."""
    from ..data_pipeline import preprocess_dataset

    instances = build_instances_from_directory(benchmark_dir, cfg=cfg, **kwargs)
    if not instances:
        raise ValueError(f"No usable trajectory instances built from {benchmark_dir}")
    return preprocess_dataset(dataset_name, instances, output_dir, cfg=cfg)
