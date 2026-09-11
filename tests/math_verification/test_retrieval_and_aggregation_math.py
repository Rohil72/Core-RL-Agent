"""Machine verification of Retrieval Metric Geometry, Kernel Weights, and Aggregation Math.

Verifies:
1. Metric space axioms (non-negativity, symmetry, identity of indiscernibles, triangle inequality).
2. Gaussian kernel properties (partition of unity, strict positivity, distance monotonicity).
3. Kish's Effective Sample Size (N_eff = 1 / sum(w^2)), proving theoretical bounds [1, K].
4. Weighted distribution statistics (mean, weighted variance, standard error, confidence intervals).
5. Information-theoretic confidence and Shannon entropy bounds in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.aggregator import (
    AggregationConfig,
    distance_weights,
    effective_sample_size,
    estimate_distribution,
)
from src.memory.confidence import (
    ConfidenceConfig,
    estimate_confidence,
    _normalised_entropy,
    _normalised_entropy_from_labels,
)
from src.memory.metric import FittedRetrievalMetric


class TestMetricGeometryAxioms:
    """Verifies that latent transformations induce valid pseudometrics or metrics."""

    def test_identity_and_diagonal_metric_axioms(self):
        """Verify metric axioms: d(x, y) >= 0, d(x, x) = 0, d(x, y) = d(y, x), d(x, z) <= d(x, y) + d(y, z)."""
        rng = np.random.default_rng(42)
        dim = 16
        mean = np.zeros(dim, dtype=np.float32)
        scale = np.ones(dim, dtype=np.float32)
        weights = rng.uniform(0.5, 2.0, size=dim).astype(np.float32)

        # Test both Identity and Diagonal metrics
        metrics = [
            FittedRetrievalMetric(kind="identity", mean=mean, scale=scale),
            FittedRetrievalMetric(kind="diagonal", mean=mean, scale=scale, parameter=weights),
        ]

        for metric in metrics:
            # Generate random points in latent space
            x = rng.normal(size=(1, dim)).astype(np.float32)
            y = rng.normal(size=(1, dim)).astype(np.float32)
            z = rng.normal(size=(1, dim)).astype(np.float32)

            tx = metric.transform(x)[0]
            ty = metric.transform(y)[0]
            tz = metric.transform(z)[0]

            # Euclidean distance in projected space
            d_xy = np.linalg.norm(tx - ty)
            d_yx = np.linalg.norm(ty - tx)
            d_xx = np.linalg.norm(tx - tx)
            d_yz = np.linalg.norm(ty - tz)
            d_xz = np.linalg.norm(tx - tz)

            # Axiom 1: Non-negativity
            assert d_xy >= 0.0

            # Axiom 2: Identity of indiscernibles
            assert np.isclose(d_xx, 0.0, atol=1e-7)

            # Axiom 3: Symmetry
            assert np.isclose(d_xy, d_yx, atol=1e-7)

            # Axiom 4: Triangle inequality: d(x, z) <= d(x, y) + d(y, z)
            assert d_xz <= d_xy + d_yz + 1e-6


class TestKernelAndAggregationMath:
    """Verifies Gaussian kernel weighting, Kish's ESS, and distribution estimates."""

    def test_gaussian_kernel_partition_of_unity_and_monotonicity(self):
        """Verify sum(w) == 1, w_i > 0, and w_i strictly decreases with distance."""
        distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=float)
        cfg = AggregationConfig(method="gaussian", gaussian_bandwidth=1.0)
        w = distance_weights(distances, cfg)

        # Invariant 1: Partition of unity (weights sum to 1.0)
        assert np.isclose(np.sum(w), 1.0, atol=1e-12)

        # Invariant 2: Strict positivity
        assert np.all(w > 0.0)

        # Invariant 3: Monotonic decrease with distance
        assert np.all(np.diff(w) < 0.0)

    def test_kish_effective_sample_size_theoretical_bounds(self):
        """Prove that 1.0 <= N_eff <= K, with equality at boundary distributions."""
        k = 25

        # 1. Theoretical Maximum: Uniform distribution w_i = 1/K -> N_eff = K
        uniform_w = np.full(k, 1.0 / k)
        assert np.isclose(effective_sample_size(uniform_w), float(k))

        # 2. Theoretical Minimum: Degenerate delta w = [1, 0, ..., 0] -> N_eff = 1.0
        delta_w = np.zeros(k)
        delta_w[0] = 1.0
        assert np.isclose(effective_sample_size(delta_w), 1.0)

        # 3. Arbitrary non-negative weights lie strictly in [1, K]
        rng = np.random.default_rng(101)
        for _ in range(50):
            raw_w = rng.uniform(0.01, 10.0, size=k)
            w = raw_w / np.sum(raw_w)
            ess = effective_sample_size(w)
            assert 1.0 - 1e-9 <= ess <= float(k) + 1e-9

    def test_distribution_estimation_and_confidence_interval(self):
        """Verify weighted mean, sample variance, standard error, and 95% CI."""
        values = np.array([-0.05, 0.02, 0.08, 0.12, 0.15], dtype=float)
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=float)
        cfg = AggregationConfig(confidence_interval_z=1.96)

        dist = estimate_distribution(values, weights, cfg)

        # Expected value: sum(w * y)
        expected_mu = float(np.sum(weights * values))
        assert np.isclose(dist.expected, expected_mu)

        # Weighted variance: sum(w * (y - mu)^2)
        expected_var = float(np.sum(weights * (values - expected_mu) ** 2))
        assert np.isclose(dist.variance, expected_var)
        assert np.isclose(dist.std, np.sqrt(expected_var))

        # Standard error = std / sqrt(N_eff)
        ess = 1.0 / np.sum(weights ** 2)
        expected_se = np.sqrt(expected_var) / np.sqrt(ess)
        expected_ci_low = expected_mu - 1.96 * expected_se
        expected_ci_high = expected_mu + 1.96 * expected_se

        assert np.isclose(dist.ci_low, expected_ci_low)
        assert np.isclose(dist.ci_high, expected_ci_high)
        assert dist.ci_low < dist.expected < dist.ci_high


class TestConfidenceAndEntropyMath:
    """Verifies agreement, disagreement, and normalized Shannon entropy bounds."""

    def test_agreement_and_disagreement_complementarity(self):
        """Verify agreement + disagreement == 1 and agreement in [0, 1]."""
        rng = np.random.default_rng(77)

        for _ in range(30):
            outcomes = rng.normal(loc=0.01, scale=0.05, size=25)
            weights = rng.uniform(0.1, 1.0, size=25)
            weights /= weights.sum()
            distances = rng.uniform(0.1, 2.0, size=25)
            tickers = np.array([f"T{i%5}" for i in range(25)])

            conf = estimate_confidence(outcomes, distances, weights, tickers)

            # Invariant 1: Agreement + Disagreement == 1.0
            assert np.isclose(conf.agreement_score + conf.disagreement_score, 1.0)

            # Invariant 2: Both scores are bounded in [0, 1]
            assert 0.0 <= conf.agreement_score <= 1.0
            assert 0.0 <= conf.disagreement_score <= 1.0
            assert 0.0 <= conf.confidence <= 1.0

    def test_shannon_entropy_bounds(self):
        """Verify normalized retrieval entropy lies strictly in [0, 1]."""
        # Maximum entropy: uniform distribution -> H = 1.0
        k = 16
        uniform_w = np.full(k, 1.0 / k)
        assert np.isclose(_normalised_entropy(uniform_w), 1.0)

        # Minimum entropy: single element -> H = 0.0
        delta_w = np.array([1.0])
        assert np.isclose(_normalised_entropy(delta_w), 0.0)

        # Label diversity entropy
        labels_diverse = np.array(["A", "B", "C", "D"])
        assert np.isclose(_normalised_entropy_from_labels(labels_diverse), 1.0)

        labels_homogeneous = np.array(["A", "A", "A", "A"])
        assert np.isclose(_normalised_entropy_from_labels(labels_homogeneous), 0.0)
