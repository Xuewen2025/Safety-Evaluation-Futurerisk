"""TRACE-replay -> RawTrajectoryInstance adapter.

Unlike `adapters/trace.py` (kinematic reconstruction from the sparse
scenario JSON, no CARLA involved), this adapter parses the OUTPUT of
actually running each TRACE `Benchmark/` scenario inside real CARLA 0.9.15
physics (`Reconstruction-Pipeline/scripts/ns_launch_scene_bm_logging.py`,
a per-tick-logging variant of the TRACE authors' own `ns_launch_scene_bm.py`
benchmark-replay script, run headless via `-nullrhi` -- no GPU rendering,
pure server-side physics/game logic, since this host's NVIDIA driver has no
GL/Vulkan userspace libraries).

Each `Benchmark/replay/replay_<crash_id>.json` holds, at CARLA's own
synchronous-mode tick rate (20Hz, `fixed_delta_seconds=1/20` in the replay
script), vehicle1/vehicle2's true simulated pose (x, y, z, yaw_deg) and
velocity (vx, vy, vz) at every tick, plus `collision_tick`: the tick index
at which CARLA's own `sensor.other.collision` sensor (attached to
vehicle1, filtered to fire only for contact specifically with vehicle2)
first registered contact -- an INDEPENDENT, non-tautological ground-truth
label, unlike `adapters/trace.py`'s geometric-overlap label (which was
close to tautological with several rule-based baselines by construction).
`status` is one of "collided" / "no_collision_route_completed" /
"no_collision_tick_cap_reached" / "spawn_failed" / "error: ...".

What's real vs. still a simplification here
----------------------------------------------
- Position, velocity, and the collision instant are genuine CARLA physics
  output (closed-loop `BasicAgent` control, real vehicle dynamics), not an
  approximation -- this is the whole point of running real CARLA instead
  of `adapters/trace.py`'s constant-speed reconstruction.
- Both vehicles are spawned as the SAME blueprint, `vehicle.tesla.model3`,
  regardless of the real crash report's actual vehicle models (this is the
  TRACE authors' own simplification in `ns_launch_scene_bm.py`, not one
  introduced here) -- so both agents get Tesla Model 3's real CARLA extent
  (`_TESLA_MODEL3_BOX_M`) and agent_class="vehicle" always; unlike
  `adapters/trace.py`, no attempt is made to parse the crash summary's
  `Model:` field into truck/bus/cyclist, since the actual simulated
  geometry never varies from a Tesla Model 3 either way.
- Heading is derived from position (`geometry.heading_from_positions`),
  the same project-wide convention as every other adapter, rather than
  trusting the logged `yaw_deg` directly (consistent with, though less
  necessary than, `adapters/bench2drive.py`'s reasoning -- CARLA's own
  transform yaw is not itself suspect here, but using the same derivation
  method as everywhere else keeps interaction-type classification and
  evasive-severity computation consistent across every adapter in this
  project).
- Collision flag: True from `collision_tick` onward (inclusive), False
  before it, for scenarios with `status == "collided"`; all-False for
  every other status (a scenario whose closed-loop BasicAgent rollout
  didn't reproduce a collision is kept as a real negative example, not
  discarded -- this is itself informative: the reconstruction's initial
  conditions didn't always reproduce contact once real physics/control
  were in the loop, unlike the by-construction-tautological label in
  `adapters/trace.py`).

Windowing: identical sliding-window mechanism to `adapters/trace.py` and
`adapters/bench2drive.py`, just at CARLA's native 20Hz (`cfg.dt=0.05`)
instead of resampled 10Hz.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import FutureRiskConfig
from ..data_pipeline import RawTrajectoryInstance
from ..geometry import BoxTrajectory, heading_from_positions

DATASET_NAME = "TRACE-replay"
REPLAY_DT = 1.0 / 20.0   # CARLA synchronous-mode fixed_delta_seconds used by ns_launch_scene_bm_logging.py

# Real CARLA `vehicle.tesla.model3` full-extent bounding box (meters):
# both vehicle1 and vehicle2 are always this blueprint (see module
# docstring) -- verified against CARLA's own blueprint library convention
# (extent is half-size; these are the doubled, full-size values), not an
# assumed/generic default the way adapters/trace.py's per-class table is.
_TESLA_MODEL3_BOX_M = (4.79, 2.16, 1.49)

REPLAY_CFG = FutureRiskConfig(dt=REPLAY_DT)


def _box_from_rows(rows: List[dict]) -> BoxTrajectory:
    x = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)
    z = np.array([r["z"] for r in rows], dtype=float)
    heading = heading_from_positions(x, y, dt=REPLAY_DT, min_speed=0.3,
                                      fallback_heading=np.radians([r["yaw_deg"] for r in rows]))
    length, width, height = _TESLA_MODEL3_BOX_M
    return BoxTrajectory(x=x, y=y, z=z, heading=heading, length=length, width=width, height=height)


def build_instances_from_replay(
    replay_path: str,
    cfg: FutureRiskConfig = REPLAY_CFG,
    window_stride_frames: Optional[int] = None,
) -> List[RawTrajectoryInstance]:
    """Build every fixed-H-second-window RawTrajectoryInstance out of one
    `Benchmark/replay/replay_<crash_id>.json` (real CARLA physics replay).
    Scenarios with status "spawn_failed" or an error are skipped entirely
    (no usable trajectory); every other status yields windows (all-negative
    for a route that completed without contact or hit the tick cap)."""
    data = json.loads(Path(replay_path).read_text())
    crash_id = data["crash_id"]
    if data["status"] in ("spawn_failed",) or data["status"].startswith("error"):
        return []
    v1_rows, v2_rows = data["vehicle1"], data["vehicle2"]
    n = min(len(v1_rows), len(v2_rows))
    if n < 2:
        return []

    ego_box = _box_from_rows(v1_rows[:n])
    agent_box = _box_from_rows(v2_rows[:n])

    collision_tick = data["collision_tick"]
    collided = data["status"] == "collided" and collision_tick is not None
    collision_flags_full = np.zeros(n, dtype=bool)
    if collided and collision_tick < n:
        collision_flags_full[collision_tick:] = True

    times_full = np.arange(n) * cfg.dt

    window_frames = int(round(cfg.horizon_s / cfg.dt)) + 1
    stride = window_stride_frames or max(window_frames // 4, 1)

    instances: List[RawTrajectoryInstance] = []
    starts = list(range(0, max(n - window_frames + 1, 0), stride))
    eff_window_frames = window_frames
    if not starts:
        starts = [0]
        eff_window_frames = n

    for w_start in starts:
        w_end = min(w_start + eff_window_frames, n)
        if w_end - w_start < 2:
            continue
        window_times = times_full[w_start:w_end] - times_full[w_start]

        def _slice(box: BoxTrajectory) -> BoxTrajectory:
            sl = slice(w_start, w_end)
            return BoxTrajectory(box.x[sl], box.y[sl], box.z[sl], box.heading[sl], box.length[sl], box.width[sl], box.height[sl])

        instances.append(RawTrajectoryInstance(
            dataset=DATASET_NAME,
            instance_id=f"tracereplay_{crash_id}__w{w_start:04d}",
            times=window_times,
            ego_box=_slice(ego_box),
            agent_boxes={"vehicle2": _slice(agent_box)},
            agent_collision_flags={"vehicle2": collision_flags_full[w_start:w_end]},
            lane_graph_available=False,
            agent_classes={"vehicle2": "vehicle"},
        ))

    return instances


def find_replay_files(root_dir: str) -> List[Path]:
    return sorted(Path(root_dir).glob("replay/replay_*.json"))


def build_instances_from_directory(
    root_dir: str, cfg: FutureRiskConfig = REPLAY_CFG, **kwargs,
) -> List[RawTrajectoryInstance]:
    instances: List[RawTrajectoryInstance] = []
    for replay_path in find_replay_files(root_dir):
        instances.extend(build_instances_from_replay(str(replay_path), cfg=cfg, **kwargs))
    return instances


def preprocess_trace_replay(
    benchmark_dir: str, output_dir: str, cfg: FutureRiskConfig = REPLAY_CFG,
    dataset_name: str = DATASET_NAME, **kwargs,
):
    from ..data_pipeline import preprocess_dataset

    instances = build_instances_from_directory(benchmark_dir, cfg=cfg, **kwargs)
    if not instances:
        raise ValueError(f"No usable trajectory instances built from {benchmark_dir}")
    return preprocess_dataset(dataset_name, instances, output_dir, cfg=cfg)
