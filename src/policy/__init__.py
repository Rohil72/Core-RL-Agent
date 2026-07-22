"""Offline entry-allocation policies that preserve the A2 exit contract."""

from src.policy.opportunity_allocator import (
    OpportunityAllocatorConfig,
    build_opportunity_features,
    fit_nonlinear_allocator,
    scores_to_allocations,
)

__all__ = [
    "OpportunityAllocatorConfig",
    "build_opportunity_features",
    "fit_nonlinear_allocator",
    "scores_to_allocations",
]

from src.policy.offline_policy import OfflinePolicyDatasetConfig, build_offline_policy_dataset

__all__ = ["OfflinePolicyDatasetConfig", "build_offline_policy_dataset"]
