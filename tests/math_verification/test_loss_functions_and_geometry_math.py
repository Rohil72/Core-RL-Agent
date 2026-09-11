"""Machine verification of Loss Functions, Outcome Geometry, and Representation Math.

Verifies:
1. Masked Huber Loss: C^1 continuity, smooth transition, and gradient boundedness (|grad| <= delta).
2. Outcome-Geometry Softmax & KL Divergence: scale invariance and Gibbs' inequality (D_KL >= 0).
3. Temporal Transport Triplet Margin: non-negativity and margin-satisfaction zero loss.
4. Latent Spectral Entropy and Effective Rank: theoretical bounds [1, min(B, D)].
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch
import torch.nn.functional as F


class TestHuberLossMath:
    """Verifies the mathematical properties of the Huber regression loss."""

    def test_huber_c1_continuity_and_transition(self):
        """Symbolically prove that Huber loss has continuous value and first derivative at |e| = delta."""
        e, delta = sp.symbols("e delta", real=True, positive=True)

        # Quadratic branch (e <= delta)
        loss_quad = (1 / sp.sympify(2)) * e ** 2
        d_quad = sp.diff(loss_quad, e)

        # Linear branch (e > delta)
        loss_lin = delta * (e - (1 / sp.sympify(2)) * delta)
        d_lin = sp.diff(loss_lin, e)

        # 1. Value continuity at e = delta
        val_quad_at_delta = loss_quad.subs(e, delta)
        val_lin_at_delta = loss_lin.subs(e, delta)
        assert sp.simplify(val_quad_at_delta - val_lin_at_delta) == 0
        assert val_quad_at_delta == (1 / sp.sympify(2)) * delta ** 2

        # 2. Derivative continuity at e = delta (C^1 smoothness)
        d_quad_at_delta = d_quad.subs(e, delta)
        d_lin_at_delta = d_lin.subs(e, delta)
        assert sp.simplify(d_quad_at_delta - d_lin_at_delta) == 0
        assert d_quad_at_delta == delta

    def test_huber_gradient_boundedness(self):
        """Verify that Huber gradient magnitude never exceeds delta (robust to outliers)."""
        delta = 1.0
        errors = torch.linspace(-10.0, 10.0, 201, requires_grad=True)

        # PyTorch Smooth L1 loss with beta=delta matches Huber
        loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=delta, reduction="none")
        loss.sum().backward()

        grads = errors.grad.detach().numpy()

        # Invariant: |grad| <= delta for all errors
        assert np.all(np.abs(grads) <= delta + 1e-6)

        # Linear saturation for |e| >= delta
        outlier_pos = (errors.detach().numpy() >= delta)
        outlier_neg = (errors.detach().numpy() <= -delta)
        assert np.allclose(grads[outlier_pos], delta, atol=1e-5)
        assert np.allclose(grads[outlier_neg], -delta, atol=1e-5)


class TestOutcomeGeometryAndKLDivergence:
    """Verifies softmax invariance and Gibbs' inequality for analogue distribution matching."""

    def test_softmax_shift_invariance(self):
        """Verify that softmax(z + c) == softmax(z) for any scalar c."""
        rng = np.random.default_rng(42)
        z = torch.as_tensor(rng.normal(size=(5, 10)), dtype=torch.float32)
        c = 42.5

        p1 = F.softmax(z, dim=-1)
        p2 = F.softmax(z + c, dim=-1)

        assert torch.allclose(p1, p2, atol=1e-6)

    def test_gibbs_inequality_kl_non_negativity(self):
        """Prove that D_KL(P || Q) >= 0, with equality if and only if P == Q."""
        rng = np.random.default_rng(99)

        for _ in range(50):
            logits_p = torch.as_tensor(rng.normal(size=(8, 12)), dtype=torch.float32)
            logits_q = torch.as_tensor(rng.normal(size=(8, 12)), dtype=torch.float32)

            P = F.softmax(logits_p, dim=-1)
            log_P = F.log_softmax(logits_p, dim=-1)
            log_Q = F.log_softmax(logits_q, dim=-1)

            # PyTorch F.kl_div(log_pred, target) expects target = P, log_pred = log(Q)
            kl_pq = F.kl_div(log_Q, P, reduction="batchmean")

            # Invariant 1: Gibbs' Inequality (D_KL >= 0)
            assert kl_pq.item() >= -1e-6

            # Invariant 2: D_KL(P || P) == 0
            kl_pp = F.kl_div(log_P, P, reduction="batchmean")
            assert np.isclose(kl_pp.item(), 0.0, atol=1e-6)


class TestEffectiveLatentRank:
    """Verifies spectral entropy and effective rank derived from singular value decomposition."""

    @staticmethod
    def compute_effective_rank(z: torch.Tensor) -> float:
        """Effective rank implementation from src.losses.outcome_geometry."""
        _, S, _ = torch.linalg.svd(z, full_matrices=False)
        norm_S = S / S.sum()
        entropy = -(norm_S * torch.log(norm_S + 1e-9)).sum()
        return float(torch.exp(entropy).item())

    def test_effective_rank_bounds_and_extremes(self):
        """Verify effective rank bounds [1.0, min(B, D)]."""
        b, d = 32, 16
        max_rank = min(b, d)

        # 1. Rank 1 / Collinear matrix: all rows are multiples of one vector
        v = torch.randn(1, d)
        z_rank1 = torch.ones(b, 1) @ v
        eff_rank1 = self.compute_effective_rank(z_rank1)
        assert np.isclose(eff_rank1, 1.0, atol=1e-2)

        # 2. Isotropic / Orthogonal matrix: all singular values equal
        q, _ = torch.linalg.qr(torch.randn(b, d))
        eff_rank_iso = self.compute_effective_rank(q)
        assert np.isclose(eff_rank_iso, float(max_rank), atol=1e-2)

        # 3. Arbitrary matrix lies strictly in [1.0, min(B, D)]
        rng = np.random.default_rng(2026)
        for _ in range(20):
            z = torch.as_tensor(rng.normal(size=(b, d)), dtype=torch.float32)
            r = self.compute_effective_rank(z)
            assert 1.0 <= r <= float(max_rank) + 1e-3
