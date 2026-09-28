"""torch Dataset/collate glue tying `features.py`'s per-instance tensors
into the padded, multi-agent `training.Batch` that `CARM`/`losses.py`
actually consume.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_CONFIG, FutureRiskConfig
from .data_pipeline import PreprocessedInstance, RawTrajectoryInstance
from .features import D_CTX, D_INT, D_KIN, build_instance_batch_item
from .training import Batch


class CARMDataset(Dataset):
    """In-memory dataset: one entry per trajectory instance, each holding
    that instance's (variable-J) agent feature tensors + labels. Instances
    with zero agents are dropped -- there's nothing for L_evt or eq:agg's
    aggregation to learn from."""

    def __init__(
        self,
        raw_instances: List[RawTrajectoryInstance],
        preprocessed_by_id: Dict[str, PreprocessedInstance],
        cfg: FutureRiskConfig = DEFAULT_CONFIG,
    ):
        self.cfg = cfg
        self.items = []
        for raw in raw_instances:
            pre = preprocessed_by_id.get(raw.instance_id)
            if pre is None or not raw.agent_boxes:
                continue
            self.items.append(build_instance_batch_item(raw, pre, cfg))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def compute_pos_weight(dataset: CARMDataset) -> float:
    """Inverse-frequency positive-class weight for L_evt's BCE term
    (S3.7's class-imbalance handling): (# agents with no event) / (# agents
    with a finite T_evt), computed once over a split (typically train)."""
    n_pos, n_neg = 0, 0
    for item in dataset.items:
        finite = np.isfinite(item["T_evt"])
        n_pos += int(finite.sum())
        n_neg += int((~finite).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def compute_risk_pos_weight(dataset: CARMDataset, pos_threshold: float = 0.5) -> float:
    """Analogous inverse-frequency weight for L_risk (see
    losses.analytic_risk_loss's docstring): (# instances with
    y_analytic <= threshold) / (# instances with y_analytic > threshold),
    computed once over a split (typically train)."""
    n_pos = sum(1 for item in dataset.items if item["y_analytic"] > pos_threshold)
    n_neg = len(dataset.items) - n_pos
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def collate_batch(
    items: List[dict], cfg: FutureRiskConfig = DEFAULT_CONFIG,
    pos_weight: float = 1.0, risk_pos_weight: float = 1.0,
) -> Batch:
    B = len(items)
    J = max(1, max(item["x_kin"].shape[0] for item in items))
    bin_edges = torch.tensor(cfg.bin_edges, dtype=torch.float32)

    x_kin = np.zeros((B, J, D_KIN), dtype=np.float32)
    X_int = np.zeros((B, J, cfg.num_bins, D_INT), dtype=np.float32)
    x_ctx = np.zeros((B, J, D_CTX), dtype=np.float32)
    tau_phy_seq = np.full((B, J, cfg.num_bins), cfg.tau_phy_clip[1], dtype=np.float32)
    T_evt = np.full((B, J), np.inf, dtype=np.float32)
    agent_mask = np.zeros((B, J), dtype=bool)
    y_analytic = np.zeros((B,), dtype=np.float32)
    collision_label = np.zeros((B,), dtype=np.int64)
    instance_ids = []

    for b, item in enumerate(items):
        n = item["x_kin"].shape[0]
        if n > 0:
            x_kin[b, :n] = item["x_kin"]
            X_int[b, :n] = item["X_int"]
            x_ctx[b, :n] = item["x_ctx"]
            tau_phy_seq[b, :n] = item["tau_phy_seq"]
            T_evt[b, :n] = item["T_evt"]
            agent_mask[b, :n] = True
        y_analytic[b] = item["y_analytic"]
        collision_label[b] = item["collision_label"]
        instance_ids.append(item["instance_id"])

    batch = Batch(
        x_kin=torch.from_numpy(x_kin),
        X_int=torch.from_numpy(X_int),
        x_ctx=torch.from_numpy(x_ctx),
        tau_phy_seq=torch.from_numpy(tau_phy_seq),
        agent_mask=torch.from_numpy(agent_mask),
        T_evt=torch.from_numpy(T_evt),
        y_analytic=torch.from_numpy(y_analytic),
        bin_edges=bin_edges,
        pos_weight=pos_weight,
        risk_pos_weight=risk_pos_weight,
    )
    batch.instance_ids = instance_ids           # extra bookkeeping, not part of the dataclass contract
    batch.collision_label = torch.from_numpy(collision_label)
    return batch


def make_collate_fn(cfg: FutureRiskConfig = DEFAULT_CONFIG, pos_weight: float = 1.0, risk_pos_weight: float = 1.0):
    return lambda items: collate_batch(items, cfg=cfg, pos_weight=pos_weight, risk_pos_weight=risk_pos_weight)
