"""Geometry-aware 3D-TTC (S3.2 / eq:3dttc).

All boxes are read directly from *observed* trajectories -- there is no
prediction here, only forward-indexing into a trajectory that has already
happened (post-hoc setting, S3.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoxTrajectory:
    """Oriented 3D box trajectory for a single agent (ego or agent j).

    All arrays have shape (T,) except `length`, `width`, `height`, which may
    be scalars or (T,) arrays (box extent is usually constant per agent but
    we allow per-frame extent for generality).
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    heading: np.ndarray       # radians
    length: np.ndarray        # along heading
    width: np.ndarray         # perpendicular to heading
    height: np.ndarray

    def __post_init__(self):
        t = len(self.x)
        for name in ("length", "width", "height"):
            val = getattr(self, name)
            if np.isscalar(val):
                setattr(self, name, np.full(t, float(val)))
        assert len(self.y) == t and len(self.z) == t and len(self.heading) == t
        assert len(self.length) == t and len(self.width) == t and len(self.height) == t

    def __len__(self):
        return len(self.x)


def rect_corners(cx: np.ndarray, cy: np.ndarray, heading: np.ndarray,
                  length: np.ndarray, width: np.ndarray) -> np.ndarray:
    """BEV rectangle corners for a batch of poses.

    Inputs broadcast to a common shape S. Returns array of shape (*S, 4, 2).
    """
    cx, cy, heading, length, width = np.broadcast_arrays(cx, cy, heading, length, width)
    hl = length / 2.0
    hw = width / 2.0
    # corners in local frame, CCW order
    local = np.stack([
        np.stack([hl, hw], axis=-1),
        np.stack([hl, -hw], axis=-1),
        np.stack([-hl, -hw], axis=-1),
        np.stack([-hl, hw], axis=-1),
    ], axis=-2)  # (*S, 4, 2)

    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.stack([
        np.stack([c, -s], axis=-1),
        np.stack([s, c], axis=-1),
    ], axis=-2)  # (*S, 2, 2)

    # rotate: (*S,4,2) @ (*S,2,2)^T  -> broadcast matmul
    world = np.einsum('...ij,...kj->...ki', rot, local)
    world[..., 0] += cx[..., None]
    world[..., 1] += cy[..., None]
    return world


def _project(corners: np.ndarray, axis: np.ndarray):
    """Project corners (*S,4,2) onto axis (*S,2); return (min, max) each (*S,)."""
    proj = np.einsum('...ki,...i->...k', corners, axis)
    return proj.min(axis=-1), proj.max(axis=-1)


def sat_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """Separating-axis test for two batches of oriented rectangles (BEV).

    corners_a, corners_b: (*S, 4, 2). Returns boolean array (*S,) -- True
    where the rectangles overlap (share interior, touching counts as no
    overlap i.e. strict intersection test uses > not >=, matching a
    conservative "occupied volume intersects" definition).
    """
    # Candidate separating axes: the two unique edge-normal directions of
    # each rectangle (rectangles only need 2 axes each, not 4, since
    # opposite edges are parallel).
    def edge_axes(corners):
        e0 = corners[..., 1, :] - corners[..., 0, :]
        e1 = corners[..., 3, :] - corners[..., 0, :]
        # normals (perpendicular), normalized (normalization not required
        # for an overlap-interval test but keeps numbers well-scaled)
        n0 = np.stack([-e0[..., 1], e0[..., 0]], axis=-1)
        n1 = np.stack([-e1[..., 1], e1[..., 0]], axis=-1)
        norm0 = np.linalg.norm(n0, axis=-1, keepdims=True)
        norm1 = np.linalg.norm(n1, axis=-1, keepdims=True)
        n0 = n0 / np.clip(norm0, 1e-9, None)
        n1 = n1 / np.clip(norm1, 1e-9, None)
        return n0, n1

    a0, a1 = edge_axes(corners_a)
    b0, b1 = edge_axes(corners_b)

    overlap = np.ones(corners_a.shape[:-2], dtype=bool)
    for axis in (a0, a1, b0, b1):
        amin, amax = _project(corners_a, axis)
        bmin, bmax = _project(corners_b, axis)
        separated = (amax < bmin) | (bmax < amin)
        overlap &= ~separated
    return overlap


def vertical_overlap(z_a: np.ndarray, h_a: np.ndarray,
                      z_b: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """Vertical-extent overlap check for full-3D occupancy (box centers z,
    heights h -> intervals [z-h/2, z+h/2])."""
    a_lo, a_hi = z_a - h_a / 2.0, z_a + h_a / 2.0
    b_lo, b_hi = z_b - h_b / 2.0, z_b + h_b / 2.0
    return (a_lo < b_hi) & (b_lo < a_hi)


def boxes_overlap(box_e: BoxTrajectory, idx_e: np.ndarray,
                   box_j: BoxTrajectory, idx_j: np.ndarray,
                   check_vertical: bool = True) -> np.ndarray:
    """Overlap test between ego frames idx_e and agent frames idx_j
    (same shape), gathering poses then running SAT (+ vertical check)."""
    corners_e = rect_corners(box_e.x[idx_e], box_e.y[idx_e], box_e.heading[idx_e],
                              box_e.length[idx_e], box_e.width[idx_e])
    corners_j = rect_corners(box_j.x[idx_j], box_j.y[idx_j], box_j.heading[idx_j],
                              box_j.length[idx_j], box_j.width[idx_j])
    overlap = sat_overlap(corners_e, corners_j)
    if check_vertical:
        overlap &= vertical_overlap(box_e.z[idx_e], box_e.height[idx_e],
                                     box_j.z[idx_j], box_j.height[idx_j])
    return overlap


def compute_3d_ttc(box_e: BoxTrajectory, box_j: BoxTrajectory,
                    dt: float = 0.1, cap: float = 5.0,
                    check_vertical: bool = True) -> np.ndarray:
    """Vectorized geometry-aware 3D-TTC, eq:3dttc.

    tau_3D^j(t) = inf{delta >= 0 : B_e(t+delta) ∩ B_j(t+delta) != empty,
                       t+delta <= H}
    with +inf if no overlap occurs within [t, H] (H is implicitly the last
    valid frame index of the trajectory, i.e. len(box_e)-1 frames at `dt`).

    Both boxes are read at their *observed* future poses (already known,
    post-hoc setting) -- this is forward *indexing*, not forward
    *simulation* of a model.

    Returns
    -------
    tau : np.ndarray, shape (T,)
        tau[t] in seconds, or np.inf if agent j never occupies the same
        volume as ego within the capped horizon.
    """
    T = len(box_e)
    assert len(box_j) == T
    n_delta = int(round(cap / dt)) + 1
    delta_frames = np.arange(n_delta)              # 0..cap/dt (inclusive)
    t_idx = np.arange(T)

    frame_idx = t_idx[:, None] + delta_frames[None, :]   # (T, n_delta)
    valid = frame_idx <= (T - 1)
    frame_idx_clipped = np.clip(frame_idx, 0, T - 1)

    overlap = boxes_overlap(box_e, frame_idx_clipped, box_j, frame_idx_clipped,
                             check_vertical=check_vertical)
    overlap &= valid

    any_overlap = overlap.any(axis=1)
    first_idx = np.argmax(overlap, axis=1)   # 0 where no True exists; guarded below
    tau = np.where(any_overlap, first_idx * dt, np.inf)
    return tau


def compute_3d_ttc_all_agents(box_e: BoxTrajectory, agent_boxes: dict,
                               dt: float = 0.1, cap: float = 5.0,
                               check_vertical: bool = True) -> dict:
    """tau_3D^j(t) for every surrounding agent j (loop over j, vectorized
    over (t, delta) within each agent -- see implementation note in
    S3.2: "vectorize across all (t, j) pairs per trajectory instance")."""
    return {
        j: compute_3d_ttc(box_e, box_j, dt=dt, cap=cap, check_vertical=check_vertical)
        for j, box_j in agent_boxes.items()
    }


def heading_from_positions(
    x: np.ndarray, y: np.ndarray, dt: float,
    min_speed: float = 0.3, fallback_heading: np.ndarray = None,
) -> np.ndarray:
    """Estimate heading (radians) from a position trace via
    atan2(vy, vx), rather than trusting a dataset's own raw orientation
    field.

    This exists because at least one real dataset adapter (Bench2Drive,
    see adapters/bench2drive.py) turned out to expose an orientation field
    whose axis convention could not be pinned down with confidence (it did
    not track the vehicle's actual direction of travel in a verifiable
    way), while a top-level "theta" was verified to be internally
    consistent for the ego vehicle only -- not for other agents, which only
    ever carry that ambiguous field. Deriving heading identically (same
    method, same convention) for *every* agent including ego, purely from
    the position trace, guarantees relative-heading computations
    (interaction.classify_interaction_type's core input) are comparing
    like with like, at the cost of being undefined for a genuinely
    stationary object (handled below).

    At low speed (< min_speed, where atan2(vy,vx) is dominated by noise)
    the heading is filled from the nearest frame (forward, then backward)
    where speed was high enough to give a meaningful direction; if the
    object never moves fast enough anywhere in the trace, `fallback_heading`
    is used (e.g. ego's own heading, a reasonable proxy for a parked
    vehicle's orientation), or 0.0 if no fallback is given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1) if fallback_heading is None else np.asarray(fallback_heading, dtype=float)[:1]

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)

    valid = speed >= min_speed
    if not valid.any():
        if fallback_heading is not None:
            return np.asarray(fallback_heading, dtype=float).copy()
        return np.zeros(n)

    # nearest-valid-neighbor fill: forward-fill then backward-fill the
    # index of the nearest frame with a trustworthy heading estimate.
    src_idx = np.where(valid, np.arange(n), -1)
    for i in range(1, n):
        if src_idx[i] == -1:
            src_idx[i] = src_idx[i - 1]
    for i in range(n - 2, -1, -1):
        if src_idx[i] == -1:
            src_idx[i] = src_idx[i + 1]
    return heading[src_idx]
