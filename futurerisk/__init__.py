"""FutureRisk: a post-hoc, outcome-agnostic metric for latent future risk,
and the Context-Adaptive Risk Model (CARM) that implements it.

Module map (mirrors the method section in instruction.txt):

    config.py          FutureRiskConfig  -- every hyperparameter, S3 defaults
    geometry.py         S3.2  Geometry-aware 3D-TTC
    interaction.py       S3.3  Interaction-type classification (Algorithm 1)
    safety_prior.py       S3.4  Analytic safe-TTC prior, margin, risk mapping
    event_proxy.py        S3.5  Harsh-evasive critical-event proxy (Algorithm 2)
    carm_model.py         S3.6  CARM: encoder, hazard head, safe-TTC head,
                                composition, bounded residual, aggregation
    losses.py             S3.7  L_risk, L_evt, L_rank, total objective
    training.py           S3.8  Two-stage training procedure
    data_pipeline.py      S3.9  Per-dataset preprocessing, caching, splits
    evaluation.py         S3.10 Cross-domain generalization evaluation
    visualization.py      S3.11 Case-study qualitative plots

Dependencies: numpy (all modules), torch (carm_model, losses, training),
scikit-learn (optional, evaluation -- falls back to a manual numpy
implementation if absent), matplotlib (visualization).
"""

from .config import FutureRiskConfig, PhysicalPriorParams, DEFAULT_CONFIG

__all__ = ["FutureRiskConfig", "PhysicalPriorParams", "DEFAULT_CONFIG"]
